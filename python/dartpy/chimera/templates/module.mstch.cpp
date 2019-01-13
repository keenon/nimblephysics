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

PYBIND11_MODULE({{module.name}}, m)
{
{{#module.bindings}}
  void {{.}}(pybind11::module& m);
  {{.}}(m);

{{/module.bindings}}
{{postcontent}}
}
{{footer}}
