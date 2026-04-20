// Jinja-based chat template renderer with tool-call support + response parser.
//
// Wraps libcommon's common_chat_templates_apply() / common_chat_parse(), which:
//   - renders the model's built-in Jinja chat template with tools / json_schema
//   - auto-generates a GBNF grammar that constrains the model's tool-call output
//     (emits lazy-grammar triggers + stops + preserved tokens; tags the output
//     with a parser "format" name)
//   - parses model-side responses back into { content, reasoning, tool_calls }
//     via a serialised PEG arena.
//
// Auto-fallback: some GGUFs store a legacy template *alias* (e.g. the literal
// string "mistral-v7-tekken") in tokenizer.chat_template instead of Jinja
// source. libcommon's Jinja renderer would treat that as a zero-interpolation
// template and return the alias verbatim — useless. We detect this case,
// fall back to llama_chat_apply_template() (the legacy C API which resolves
// aliases), and return the plain rendered prompt. If tools/jsonSchema were
// supplied we throw — the legacy path cannot honour them, and silent drops
// produced real bugs during development.

#include "llama_model.h"

#include "chat.h"
#include "common.h"
#include "peg-parser.h"

#include <nlohmann/json.hpp>

#include <exception>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Convert libcommon's typed triggers into the (patterns[], tokens[]) pair
// accepted by llama_sampler_init_grammar_lazy_patterns().
// Uses common/common.h's regex_escape() for COMMON_GRAMMAR_TRIGGER_TYPE_WORD.
static void flatten_triggers(
    const std::vector<common_grammar_trigger> & triggers,
    std::vector<std::string>                  & out_patterns,
    std::vector<llama_token>                  & out_tokens)
{
    for (const auto & t : triggers) {
        switch (t.type) {
            case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                out_patterns.push_back(regex_escape(t.value));
                break;
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                out_patterns.push_back(t.value);
                break;
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                std::string p = t.value;
                if (p.empty() || (p.front() != '^' && p.back() != '$')) {
                    p = std::string("^") + t.value + (!t.value.empty() && t.value.back() == '$' ? "" : "$");
                }
                out_patterns.push_back(std::move(p));
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                out_tokens.push_back(t.token);
                break;
        }
    }
}

static common_chat_tool_choice parse_tool_choice(const std::string & s) {
    if (s == "required") return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    if (s == "none")     return COMMON_CHAT_TOOL_CHOICE_NONE;
    return COMMON_CHAT_TOOL_CHOICE_AUTO;
}

// Heuristic: is this template string full Jinja, or just a legacy alias name
// (e.g. "mistral-v7-tekken")? Full Jinja always contains at least one `{%` or
// `{{`; aliases are short, alphanumeric with dashes.
static bool looks_like_jinja(const std::string & tmpl) {
    return tmpl.find("{%") != std::string::npos ||
           tmpl.find("{{") != std::string::npos;
}

// Render via llama_chat_apply_template() (the legacy C API that resolves
// named aliases). Used as a fallback when the embedded template is not Jinja.
static std::string render_legacy(
    llama_model * model,
    const json &  messages_json,
    bool          add_generation_prompt,
    std::string & error_out)
{
    const char * tmpl = llama_model_chat_template(model, nullptr);

    std::vector<std::string> roles;
    std::vector<std::string> contents;
    std::vector<llama_chat_message> chat;
    roles.reserve(messages_json.size());
    contents.reserve(messages_json.size());
    chat.reserve(messages_json.size());

    for (const auto & m : messages_json) {
        if (!m.contains("role") || !m.contains("content")) {
            error_out = "legacy fallback: each message must have role + content (string)";
            return {};
        }
        const auto & c = m["content"];
        if (!c.is_string()) {
            error_out = "legacy fallback: content must be a plain string (legacy path has no tool/multipart support)";
            return {};
        }
        roles.push_back(m["role"].get<std::string>());
        contents.push_back(c.get<std::string>());
    }
    for (size_t i = 0; i < roles.size(); ++i) {
        chat.push_back({ roles[i].c_str(), contents[i].c_str() });
    }

    int32_t len = llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), add_generation_prompt, nullptr, 0);
    if (len < 0) {
        error_out = "llama_chat_apply_template: unsupported template";
        return {};
    }
    std::vector<char> buf((size_t)len + 1);
    llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), add_generation_prompt, buf.data(), buf.size());
    return std::string(buf.data(), (size_t)len);
}

// Map "peg-native" / "peg-simple" / "peg-gemma4" / "Content-only" back to
// the enum. Only four values exist — simple switch.
static common_chat_format parse_format_name(const std::string & name) {
    if (name == "Content-only") return COMMON_CHAT_FORMAT_CONTENT_ONLY;
    if (name == "peg-simple")   return COMMON_CHAT_FORMAT_PEG_SIMPLE;
    if (name == "peg-native")   return COMMON_CHAT_FORMAT_PEG_NATIVE;
    if (name == "peg-gemma4")   return COMMON_CHAT_FORMAT_PEG_GEMMA4;
    throw std::runtime_error("Unknown chat format name: " + name);
}

// ---------------------------------------------------------------------------
// applyChatTemplateJinja(messagesJson, toolsJson, opts)
// ---------------------------------------------------------------------------
//
//   info[0] : messages JSON string   (OAI-compat array)
//   info[1] : tools JSON string      (OAI-compat array; "[]" if none)
//   info[2] : options object
//
// Returns:
//   { prompt, format,
//     parser?,                          -- serialised PEG arena (opaque blob)
//     grammar?, grammarLazy?,
//     grammarTriggerPatterns?, grammarTriggerTokens?,
//     preservedTokens?, additionalStops? }
//
// On alias-only templates, returns:
//   { prompt, format: "legacy" }
// and throws if tools/jsonSchema were supplied.
//
Napi::Value LlamaModel::ApplyChatTemplateJinja(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (!model_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (info.Length() < 3 ||
        !info[0].IsString() ||
        !info[1].IsString() ||
        !info[2].IsObject())
    {
        Napi::TypeError::New(env,
            "applyChatTemplateJinja(messagesJson: string, toolsJson: string, opts: object)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    const std::string messages_str = info[0].As<Napi::String>().Utf8Value();
    const std::string tools_str    = info[1].As<Napi::String>().Utf8Value();
    Napi::Object      opts         = info[2].As<Napi::Object>();

    std::string template_override;
    if (opts.Has("chatTemplateOverride") && opts.Get("chatTemplateOverride").IsString()) {
        template_override = opts.Get("chatTemplateOverride").As<Napi::String>().Utf8Value();
    }

    const bool has_tools =
        !tools_str.empty() && tools_str != "[]" && tools_str != "null";
    const bool has_schema =
        opts.Has("jsonSchema") && opts.Get("jsonSchema").IsString() &&
        !opts.Get("jsonSchema").As<Napi::String>().Utf8Value().empty();

    // ---- Detect alias-only template → legacy fallback path ----
    if (template_override.empty()) {
        const char * embedded = llama_model_chat_template(model_, nullptr);
        if (!embedded || !looks_like_jinja(embedded)) {
            if (has_tools || has_schema) {
                Napi::Error::New(env,
                    "applyChatTemplateJinja: model's embedded template is a legacy "
                    "alias; tools / jsonSchema cannot be honoured on this path. "
                    "Provide `chatTemplateOverride` with full Jinja source, or "
                    "call applyChatTemplate() (legacy) without tools.")
                    .ThrowAsJavaScriptException();
                return env.Undefined();
            }

            bool add_gen_prompt = true;
            if (opts.Has("addGenerationPrompt") && opts.Get("addGenerationPrompt").IsBoolean()) {
                add_gen_prompt = opts.Get("addGenerationPrompt").As<Napi::Boolean>().Value();
            }

            json messages_json;
            try {
                messages_json = json::parse(messages_str);
            } catch (const std::exception & e) {
                Napi::Error::New(env, std::string("messages JSON parse failed: ") + e.what())
                    .ThrowAsJavaScriptException();
                return env.Undefined();
            }

            std::string err;
            std::string prompt = render_legacy(model_, messages_json, add_gen_prompt, err);
            if (!err.empty()) {
                Napi::Error::New(env, err).ThrowAsJavaScriptException();
                return env.Undefined();
            }

            Napi::Object result = Napi::Object::New(env);
            result.Set("prompt", Napi::String::New(env, prompt));
            result.Set("format", Napi::String::New(env, "legacy"));
            return result;
        }
    }

    // ---- Lazy-init the templates bundle (possibly with override) ----
    if (!chat_templates_) {
        auto tmpls = common_chat_templates_init(model_, template_override);
        if (!tmpls) {
            Napi::Error::New(env, "common_chat_templates_init failed")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }
        chat_templates_ = tmpls.release();  // freed in dtor/Dispose
    }

    common_chat_templates_inputs inputs;

    try {
        const json messages_json = json::parse(messages_str);
        inputs.messages = common_chat_msgs_parse_oaicompat(messages_json);

        if (has_tools) {
            const json tools_json = json::parse(tools_str);
            inputs.tools = common_chat_tools_parse_oaicompat(tools_json);
        }
    } catch (const std::exception & e) {
        Napi::Error::New(env,
            std::string("Failed to parse messages/tools JSON: ") + e.what())
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // ---- Options ----
    inputs.use_jinja             = true;
    inputs.add_generation_prompt = true;
    inputs.parallel_tool_calls   = false;
    inputs.enable_thinking       = true;
    inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_AUTO;

    if (opts.Has("addGenerationPrompt") && opts.Get("addGenerationPrompt").IsBoolean()) {
        inputs.add_generation_prompt = opts.Get("addGenerationPrompt").As<Napi::Boolean>().Value();
    }
    if (opts.Has("parallelToolCalls") && opts.Get("parallelToolCalls").IsBoolean()) {
        inputs.parallel_tool_calls = opts.Get("parallelToolCalls").As<Napi::Boolean>().Value();
    }
    if (opts.Has("enableThinking") && opts.Get("enableThinking").IsBoolean()) {
        inputs.enable_thinking = opts.Get("enableThinking").As<Napi::Boolean>().Value();
    }
    if (opts.Has("toolChoice") && opts.Get("toolChoice").IsString()) {
        inputs.tool_choice = parse_tool_choice(
            opts.Get("toolChoice").As<Napi::String>().Utf8Value());
    }
    if (opts.Has("grammar") && opts.Get("grammar").IsString()) {
        inputs.grammar = opts.Get("grammar").As<Napi::String>().Utf8Value();
    }
    if (opts.Has("jsonSchema") && opts.Get("jsonSchema").IsString()) {
        inputs.json_schema = opts.Get("jsonSchema").As<Napi::String>().Utf8Value();
    }
    if (opts.Has("chatTemplateKwargs") && opts.Get("chatTemplateKwargs").IsObject()) {
        Napi::Object kw = opts.Get("chatTemplateKwargs").As<Napi::Object>();
        Napi::Array  keys = kw.GetPropertyNames();
        for (uint32_t i = 0; i < keys.Length(); ++i) {
            std::string k = keys.Get(i).As<Napi::String>().Utf8Value();
            Napi::Value v = kw.Get(k);
            if (v.IsString()) {
                inputs.chat_template_kwargs[k] = v.As<Napi::String>().Utf8Value();
            }
        }
    }

    // ---- Apply ----
    common_chat_params params;
    try {
        params = common_chat_templates_apply(chat_templates_, inputs);
    } catch (const std::exception & e) {
        Napi::Error::New(env,
            std::string("common_chat_templates_apply failed: ") + e.what())
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // ---- Build return object ----
    Napi::Object result = Napi::Object::New(env);
    result.Set("prompt", Napi::String::New(env, params.prompt));
    result.Set("format", Napi::String::New(env, common_chat_format_name(params.format)));
    if (!params.parser.empty()) {
        result.Set("parser", Napi::String::New(env, params.parser));
    }
    if (!params.generation_prompt.empty()) {
        result.Set("generationPrompt", Napi::String::New(env, params.generation_prompt));
    }

    if (!params.grammar.empty()) {
        result.Set("grammar",     Napi::String::New(env, params.grammar));
        result.Set("grammarLazy", Napi::Boolean::New(env, params.grammar_lazy));

        if (!params.grammar_triggers.empty()) {
            std::vector<std::string> pats;
            std::vector<llama_token> toks;
            flatten_triggers(params.grammar_triggers, pats, toks);

            Napi::Array pats_arr = Napi::Array::New(env, pats.size());
            for (size_t i = 0; i < pats.size(); ++i) {
                pats_arr.Set((uint32_t)i, Napi::String::New(env, pats[i]));
            }
            Napi::Array toks_arr = Napi::Array::New(env, toks.size());
            for (size_t i = 0; i < toks.size(); ++i) {
                toks_arr.Set((uint32_t)i, Napi::Number::New(env, (double)toks[i]));
            }
            result.Set("grammarTriggerPatterns", pats_arr);
            result.Set("grammarTriggerTokens",  toks_arr);
        }
    }

    if (!params.preserved_tokens.empty()) {
        Napi::Array arr = Napi::Array::New(env, params.preserved_tokens.size());
        for (size_t i = 0; i < params.preserved_tokens.size(); ++i) {
            arr.Set((uint32_t)i, Napi::String::New(env, params.preserved_tokens[i]));
        }
        result.Set("preservedTokens", arr);
    }

    if (!params.additional_stops.empty()) {
        Napi::Array arr = Napi::Array::New(env, params.additional_stops.size());
        for (size_t i = 0; i < params.additional_stops.size(); ++i) {
            arr.Set((uint32_t)i, Napi::String::New(env, params.additional_stops[i]));
        }
        result.Set("additionalStops", arr);
    }

    return result;
}

// ---------------------------------------------------------------------------
// parseChatResponse(text, { format, parser?, generationPrompt?, isPartial? })
// ---------------------------------------------------------------------------
//
// Round-trips the opaque `parser` blob from applyChatTemplateJinja() and
// returns { content, reasoningContent, toolCalls: [{name, arguments, id}] }.
// Works for all formats emitted by libcommon; the legacy fallback path
// (format === "legacy") can't be parsed this way, so we return content-only.
//
Napi::Value LlamaModel::ParseChatResponse(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsObject()) {
        Napi::TypeError::New(env,
            "parseChatResponse(text: string, opts: { format, parser?, generationPrompt?, isPartial? })")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    const std::string text = info[0].As<Napi::String>().Utf8Value();
    Napi::Object      opts = info[1].As<Napi::Object>();

    if (!opts.Has("format") || !opts.Get("format").IsString()) {
        Napi::TypeError::New(env, "parseChatResponse: opts.format (string) is required")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    const std::string format_name = opts.Get("format").As<Napi::String>().Utf8Value();

    // Legacy-path outputs have no parser; return content as-is.
    if (format_name == "legacy") {
        Napi::Object msg = Napi::Object::New(env);
        msg.Set("content",          Napi::String::New(env, text));
        msg.Set("reasoningContent", Napi::String::New(env, ""));
        msg.Set("toolCalls",        Napi::Array::New(env, 0));
        return msg;
    }

    common_chat_parser_params pparams;
    try {
        pparams.format = parse_format_name(format_name);
    } catch (const std::exception & e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (opts.Has("generationPrompt") && opts.Get("generationPrompt").IsString()) {
        pparams.generation_prompt = opts.Get("generationPrompt").As<Napi::String>().Utf8Value();
    }
    if (opts.Has("parseToolCalls") && opts.Get("parseToolCalls").IsBoolean()) {
        pparams.parse_tool_calls = opts.Get("parseToolCalls").As<Napi::Boolean>().Value();
    }
    bool is_partial = false;
    if (opts.Has("isPartial") && opts.Get("isPartial").IsBoolean()) {
        is_partial = opts.Get("isPartial").As<Napi::Boolean>().Value();
    }

    // Deserialise the PEG arena from the opaque blob produced by apply().
    // Empty string → libcommon falls back to a pure-content parser.
    if (opts.Has("parser") && opts.Get("parser").IsString()) {
        const std::string parser_blob = opts.Get("parser").As<Napi::String>().Utf8Value();
        if (!parser_blob.empty()) {
            try {
                pparams.parser.load(parser_blob);
            } catch (const std::exception & e) {
                Napi::Error::New(env,
                    std::string("parser blob deserialisation failed: ") + e.what())
                    .ThrowAsJavaScriptException();
                return env.Undefined();
            }
        }
    }

    common_chat_msg msg;
    try {
        msg = common_chat_parse(text, is_partial, pparams);
    } catch (const std::exception & e) {
        Napi::Error::New(env, std::string("common_chat_parse failed: ") + e.what())
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // ---- Build return object ----
    Napi::Object out = Napi::Object::New(env);
    out.Set("content",          Napi::String::New(env, msg.content));
    out.Set("reasoningContent", Napi::String::New(env, msg.reasoning_content));

    Napi::Array calls = Napi::Array::New(env, msg.tool_calls.size());
    for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
        Napi::Object c = Napi::Object::New(env);
        c.Set("name",      Napi::String::New(env, msg.tool_calls[i].name));
        c.Set("arguments", Napi::String::New(env, msg.tool_calls[i].arguments));
        c.Set("id",        Napi::String::New(env, msg.tool_calls[i].id));
        calls.Set((uint32_t)i, c);
    }
    out.Set("toolCalls", calls);

    if (!msg.tool_name.empty())    out.Set("toolName",   Napi::String::New(env, msg.tool_name));
    if (!msg.tool_call_id.empty()) out.Set("toolCallId", Napi::String::New(env, msg.tool_call_id));

    return out;
}
