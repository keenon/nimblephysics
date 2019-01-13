{{header}}
{{#includes}}
#include <{{.}}>
{{/includes}}
{{#sources}}
#include <{{.}}>
{{/sources}}
{{precontent}}
#include <pybind11/pybind11.h>
{{postinclude}}

void {{variable.mangled_name}}(::pybind11::module& m)
{
    auto sm = m{{!
        }}{{#variable.namespace_scope}}{{#name}}.def_submodule("{{name}}"){{/name}}{{/variable.namespace_scope}};

    auto attr = sm{{!
        }}{{#variable.class_scope}}{{#name}}.attr("{{name}}"){{/name}}{{/variable.class_scope}};

    attr.attr("{{variable.name}}") = {{variable.qualified_name}};
}

{{postcontent}}
{{footer}}
