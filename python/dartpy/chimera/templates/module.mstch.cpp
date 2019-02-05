{{header}}
{{#includes}}
#include <{{.}}>
{{/includes}}
{{#sources}}
#include <{{.}}>
{{/sources}}
#include <iostream>
#include <pybind11/pybind11.h>
{{postinclude}}

{{#module.bindings}}
void {{.}}(pybind11::module& m);
{{/module.bindings}}

PYBIND11_MODULE({{module.name}}, m)
{
    std::cout << "[Debug] Loading module '" << "{{function.mangled_name}}" << "'" << std::endl;

    {{precontent}}

{{#module.bindings}}
    try {
      {{.}}(m);
    } catch(...) {
      std::cerr << "Exception in: {{.}}" << std::endl;
      throw;
    }
{{/module.bindings}}

    {{postcontent}}
}
{{footer}}
