export type CreateBoxCommand = {
  type: "create_box";
  key: string;
  size: number[];
  pos: number[];
  euler: number[];
  color: number[];
  cast_shadows: boolean;
  receive_shadows: boolean;
};

export type CreateSphereCommand = {
  type: "create_sphere";
  key: string;
  radius: number;
  pos: number[];
  color: number[];
  cast_shadows: boolean;
  receive_shadows: boolean;
};

export type CreateCapsuleCommand = {
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

export type CreateLineCommand = {
  type: "create_line";
  key: string;
  points: number[][];
  color: number[];
};

export type CreateMeshCommand = {
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

export type CreateTextureCommand = {
  type: "create_texture";
  key: string;
  base64: string;
};

export type SetObjectPosCommand = {
  type: "set_object_pos";
  key: string;
  pos: number[];
};

export type SetObjectRotationCommand = {
  type: "set_object_rotation";
  key: string;
  euler: number[];
};

export type DeleteObjectCommand = {
  type: "delete_object";
  key: string;
};

export type SetObjectColorCommand = {
  type: "set_object_color";
  key: string;
  color: number[];
};

export type SetObjectScaleCommand = {
  type: "set_object_scale";
  key: string;
  scale: number[];
};

export type EnableMouseInteractionCommand = {
  type: "enable_mouse";
  key: string;
};

export type DisableMouseInteractionCommand = {
  type: "disable_mouse";
  key: string;
};

export type CreateTextCommand = {
  type: "create_text";
  key: string;
  from_top_left: number[];
  size: number[];
  contents: string;
};

export type CreateButtonCommand = {
  type: "create_button";
  key: string;
  from_top_left: number[];
  size: number[];
  label: string;
};

export type CreateSliderCommand = {
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

export type CreatePlotCommand = {
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

export type SetUIElementPositionCommmand = {
  type: "set_ui_elem_pos";
  key: string;
  from_top_left: number[];
};

export type SetUIElementSizeCommmand = {
  type: "set_ui_elem_size";
  key: string;
  size: number[];
};

export type DeleteUIElementCommmand = {
  type: "delete_ui_elem";
  key: string;
};

export type SetTextContents = {
  type: "set_text_contents";
  key: string;
  contents: string;
};

export type SetButtonLabel = {
  type: "set_button_label";
  key: string;
  label: string;
};

export type SetSliderValue = {
  type: "set_slider_value";
  key: string;
  value: number;
};

export type SetSliderMin = {
  type: "set_slider_min";
  key: string;
  min: number;
};

export type SetSliderMax = {
  type: "set_slider_max";
  key: string;
  max: number;
};

export type SetPlotData = {
  type: "set_plot_data";
  key: string;
  min_x: number;
  max_x: number;
  xs: number[];
  min_y: number;
  max_y: number;
  ys: number[];
};

export type Command =
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

export type CommandRecording = Command[][];
