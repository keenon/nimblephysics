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

namespace {

{{#class.comment?}}{{!
}}constexpr char {{class.mangled_name}}_docstring[] = R"CHIMERA_STRING({{!
}}{{#class.comment}}{{!
}}{{.}}
{{/class.comment}}{{!
}})CHIMERA_STRING";
{{/class.comment?}}

{{#class.methods}}{{!
}}{{#comment?}}{{!
}}constexpr char {{mangled_name}}_docstring[] = R"CHIMERA_STRING({{!
}}{{#comment}}{{!
}}{{.}}
{{/comment}}{{!
}})CHIMERA_STRING";

{{/comment?}}{{!
}}{{/class.methods}}

} // namespace

void {{class.mangled_name}}(pybind11::module& m)
{
    auto sm = m{{!
        }}{{#class.namespace_scope}}{{#name}}.def_submodule("{{name}}"){{/name}}{{/class.namespace_scope}};

    auto attr = sm{{!
        }}{{#class.class_scope}}{{#name}}.attr("{{name}}"){{/name}}{{/class.class_scope}};

    ::pybind11::class_<{{class.type}}{{!
        }}{{#class.held_type}}, {{!
        }}{{.}}{{/class.held_type}}{{#class.bases?}}, {{!
        }}{{!
            }}{{#class.bases}}{{qualified_name}}{{^last}}, {{/last}}{{/class.bases}}{{!
        }}{{/class.bases?}} >(attr, "{{class.name}}"{{!
        }}{{#class.comment?}}, {{class.mangled_name}}_docstring{{/class.comment?}}{{!
        }}){{!

/* constructors */}}
{{#class.constructors}}{{!
}}{{#overloads}}{{!
    }}        .def(::pybind11::init{{!
        }}<{{#params}}{{type}}{{^last}}, {{/last}}{{/params}}>(){{!
        }}{{#params?}}, {{#params}}::pybind11::arg("{{name}}"){{^last}}, {{/last}}{{/params}}{{/params?}})
{{/overloads}}{{!
}}{{/class.constructors}}{{!

/* member functions */}}
{{#class.methods}}{{!
}}{{^is_static}}{{!
}}{{#overloads}}{{!
    }}        .def("{{name}}", +[]({{#is_const}}const {{/is_const}}{{class.type}} *self{{#params}}, {{type}} {{name}}{{/params}}){{!
    }}{{#is_void}} { {{/is_void}}{{!
    }}{{^is_void}} -> {{return_type}} { return {{/is_void}}{{!
    }}self->{{call}}({{#params}}{{name}}{{^last}}, {{/last}}{{/params}}); }{{!
    }}{{#return_value_policy}}, ::pybind11::return_value_policy::{{.}}{{/return_value_policy}}{{!
    }}{{#comment?}}, {{mangled_name}}_docstring{{/comment?}}{{!
    }}{{#params?}}, {{#params}}::pybind11::arg("{{name}}"){{^last}}, {{/last}}{{/params}}{{/params?}})
{{/overloads}}{{!
}}{{/is_static}}{{!
}}{{/class.methods}}{{!

/* static member functions */}}
{{#class.methods}}{{!
}}{{#is_static}}{{!
}}{{#overloads}}{{!
    }}        .def_static("{{name}}", +[]({{#params}}{{type}} {{name}}{{^last}}, {{/last}}{{/params}}){{!
    }}{{#is_void}} { {{/is_void}}{{!
    }}{{^is_void}} -> {{return_type}} { return {{/is_void}}{{!
    }}{{qualified_call}}({{#params}}{{name}}{{^last}}, {{/last}}{{/params}}); }{{!
    }}{{#return_value_policy}}, ::pybind11::return_value_policy::{{.}}{{/return_value_policy}}{{!
    }}{{#params?}}, {{#params}}::pybind11::arg("{{name}}"){{^last}}, {{/last}}{{/params}}{{/params?}})
{{/overloads}}{{!
}}{{/is_static}}{{!
}}{{/class.methods}}{{!

/* fields */}}
{{#class.fields}}{{!
    }}{{#is_assignable}}        .def_readwrite{{/is_assignable}}{{!
    }}{{^is_assignable}}        .def_readonly{{/is_assignable}}{{!
    }}("{{name}}", &{{qualified_name}})
{{/class.fields}}{{!

/* static fields */}}
{{#class.static_fields}}{{!
    }}{{#is_assignable}}        .def_readwrite_static{{/is_assignable}}{{!
    }}{{^is_assignable}}        .def_readonly_static{{/is_assignable}}{{!
    }}("{{name}}", &{{qualified_name}})
{{/class.static_fields}}
    ;
}
{{postcontent}}
{{footer}}
