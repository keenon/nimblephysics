{{header}}
{{#includes}}
#include <{{.}}>
{{/includes}}
{{#sources}}
#include <{{.}}>
{{/sources}}
#include <pybind11/pybind11.h>
{{postinclude}}

{{#module.bindings}}
void {{.}}(pybind11::module& m);
{{/module.bindings}}

PYBIND11_MODULE({{module.name}}, m)
{
{{precontent}}
{{#module.bindings}}
    {{.}}(m);
{{/module.bindings}}
{{postcontent}}
}
{{footer}}