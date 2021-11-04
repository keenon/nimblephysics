type CreateBoxCommand = {
  type: "create_box";
  key: string;
  size: number[];
  pos: number[];
  euler: number[];
  color: number[];
  cast_shadows: boolean;
  receive_shadows: boolean;
};

type CreateSphereCommand = {
  type: "create_sphere";
  key: string;
  radius: number;
  pos: number[];
  color: number[];
  cast_shadows: boolean;
  receive_shadows: boolean;
};

type CreateCapsuleCommand = {
  type: "create_capsule";
  key: string;
  radius: number;
  height: number;
  pos: number[];
  euler: number[];
  color: number[];
  cast_shadows: boolean;
  receive_shadows: boolean;
};

type CreateLineCommand = {
  type: "create_line";
  key: string;
  points: number[][];
  color: number[];
};

type CreateMeshCommand = {
  type: "create_mesh";
  key: string;
  vertices: number[][];
  vertex_normals: number[][];
  faces: number[][];
  uv: number[][];
  texture_starts: { key: string; start: number }[];
  pos: number[];
  euler: number[];
  scale: number[];
  color: number[];
  cast_shadows: boolean;
  receive_shadows: boolean;
};

type CreateTextureCommand = {
  type: "create_texture";
  key: string;
  base64: string;
};

type SetObjectPosCommand = {
  type: "set_object_pos";
  key: string;
  pos: number[];
};

type SetObjectRotationCommand = {
  type: "set_object_rotation";
  key: string;
  euler: number[];
};

type DeleteObjectCommand = {
  type: "delete_object";
  key: string;
};

type SetObjectColorCommand = {
  type: "set_object_color";
  key: string;
  color: number[];
};

type SetObjectScaleCommand = {
  type: "set_object_scale";
  key: string;
  scale: number[];
};

type EnableMouseInteractionCommand = {
  type: "enable_mouse";
  key: string;
};

type DisableMouseInteractionCommand = {
  type: "disable_mouse";
  key: string;
};

type CreateTextCommand = {
  type: "create_text";
  key: string;
  from_top_left: number[];
  size: number[];
  contents: string;
};

type CreateButtonCommand = {
  type: "create_button";
  key: string;
  from_top_left: number[];
  size: number[];
  label: string;
};

type CreateSliderCommand = {
  type: "create_slider";
  key: string;
  from_top_left: number[];
  size: number[];
  min: number;
  max: number;
  value: number;
  only_ints: boolean;
  horizontal: boolean;
};

type CreatePlotCommand = {
  type: "create_plot";
  key: string;
  from_top_left: number[];
  size: number[];
  min_x: number;
  max_x: number;
  xs: number[];
  min_y: number;
  max_y: number;
  ys: number[];
  plot_type: "line" | "scatter";
};

type SetUIElementPositionCommmand = {
  type: "set_ui_elem_pos";
  key: string;
  from_top_left: number[];
};

type SetUIElementSizeCommmand = {
  type: "set_ui_elem_size";
  key: string;
  size: number[];
};

type DeleteUIElementCommmand = {
  type: "delete_ui_elem";
  key: string;
};

type SetTextContents = {
  type: "set_text_contents";
  key: string;
  contents: string;
};

type SetButtonLabel = {
  type: "set_button_label";
  key: string;
  label: string;
};

type SetSliderValue = {
  type: "set_slider_value";
  key: string;
  value: number;
};

type SetSliderMin = {
  type: "set_slider_min";
  key: string;
  min: number;
};

type SetSliderMax = {
  type: "set_slider_max";
  key: string;
  max: number;
};

type SetPlotData = {
  type: "set_plot_data";
  key: string;
  min_x: number;
  max_x: number;
  xs: number[];
  min_y: number;
  max_y: number;
  ys: number[];
};

type Command =
  | CreateBoxCommand
  | CreateSphereCommand
  | CreateCapsuleCommand
  | CreateLineCommand
  | CreateMeshCommand
  | CreateTextureCommand
  | SetObjectPosCommand
  | SetObjectRotationCommand
  | SetObjectColorCommand
  | SetObjectScaleCommand
  | DeleteObjectCommand
  | EnableMouseInteractionCommand
  | DisableMouseInteractionCommand
  | CreateTextCommand
  | CreateButtonCommand
  | CreateSliderCommand
  | CreatePlotCommand
  | SetUIElementPositionCommmand
  | SetUIElementSizeCommmand
  | DeleteUIElementCommmand
  | SetTextContents
  | SetButtonLabel
  | SetSliderValue
  | SetSliderMin
  | SetSliderMax
  | SetPlotData;

type CommandRecording = Command[][];
