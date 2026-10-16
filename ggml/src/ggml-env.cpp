// halo-hybrid: see ggml-env.h
#include "ggml-env.h"
#include "ggml-impl.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

#if defined(_WIN32)
#define GGML_ENVIRON _environ
#else
extern char ** environ;
#define GGML_ENVIRON environ
#endif

static std::mutex            g_env_mutex;
static std::set<std::string> g_env_warned;   // names already warned about (unparsable value, unsafe switch)

static const char * env_get(const char * name) {
    return getenv(name);
}

static void env_warn_once(const char * name, const char * value, const char * what) {
    std::lock_guard<std::mutex> lock(g_env_mutex);
    if (g_env_warned.insert(name).second) {
        GGML_LOG_WARN("%s=%s is not %s; using the default\n", name, value, what);
    }
}

static bool env_eq_nocase(const char * a, const char * b) {
    for (; *a && *b; ++a, ++b) {
        if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
            return false;
        }
    }
    return *a == *b;
}

bool ggml_env_flag(const char * name, bool def) {
    const char * v = env_get(name);
    if (v == nullptr) {
        return def;
    }
    for (const char * off : { "", "0", "false", "off", "no" }) {
        if (env_eq_nocase(v, off)) {
            return false;
        }
    }
    return true;
}

bool ggml_env_flag_unsafe(const char * name, bool def, const char * risk) {
    const bool v = ggml_env_flag(name, def);
    if (v != def) {
        std::lock_guard<std::mutex> lock(g_env_mutex);
        if (g_env_warned.insert(std::string(name) + "#unsafe").second) {
            GGML_LOG_WARN("%s=%s is a diagnostic switch that %s; do not use it for normal runs\n", name, env_get(name), risk);
        }
    }
    return v;
}

int64_t ggml_env_int(const char * name, int64_t def) {
    const char * v = env_get(name);
    if (v == nullptr) {
        return def;
    }
    char * end = nullptr;
    errno = 0;
    const long long r = std::strtoll(v, &end, 10);
    while (end && std::isspace((unsigned char) *end)) {
        ++end;
    }
    if (end == v || *end != '\0' || errno == ERANGE) {
        env_warn_once(name, v, "an integer");
        return def;
    }
    return (int64_t) r;
}

double ggml_env_float(const char * name, double def) {
    const char * v = env_get(name);
    if (v == nullptr) {
        return def;
    }
    char * end = nullptr;
    errno = 0;
    const double r = std::strtod(v, &end);
    while (end && std::isspace((unsigned char) *end)) {
        ++end;
    }
    if (end == v || *end != '\0' || errno == ERANGE) {
        env_warn_once(name, v, "a number");
        return def;
    }
    return r;
}

const char * ggml_env_set_string(void) {
    static const std::string set = [] {
        std::string r;
        for (char ** e = GGML_ENVIRON; e && *e; ++e) {
            // LLAMA_ARG_* mirrors command-line options (--api-key among them); never log those
            if ((std::strncmp(*e, "GGML_", 5) != 0 && std::strncmp(*e, "LLAMA_", 6) != 0) || std::strncmp(*e, "LLAMA_ARG_", 10) == 0) {
                continue;
            }
            std::string kv = *e;
            const std::string name = kv.substr(0, kv.find('='));
            if (name.find("KEY") != std::string::npos || name.find("TOKEN") != std::string::npos ||
                    name.find("SECRET") != std::string::npos || name.find("PASS") != std::string::npos) {
                kv = name + "=<redacted>";
            }
            r += " " + kv;
        }
        return r;
    }();
    return set.c_str();
}

void ggml_env_log_set(const char * who) {
    static std::once_flag once;
    std::call_once(once, [who] {
        const char * set = ggml_env_set_string();
        if (set[0] != '\0') {
            GGML_LOG_INFO("%s: environment:%s\n", who, set);
        }
    });
}
