/**
 * ============================================================
 *  模具渲染着色器 (moldRender.wgsl)
 * ============================================================
 *
 *  渲染模具的三角网格外表面
 *
 *  光照模型：纯 Lambert 哑光
 *    ambient = 0.32
 *    diffuse = max(n·l, 0) × 0.55
 *
 *  渲染特性：
 *    - 半透明（alpha = 0.2），可以看到内部颗粒
 *    - 写入表面深度，只显示当前可见面的有限元网格线
 *    - 双面渲染（cullMode: none）
 *
 *  Uniform 布局（256 字节，全部使用 vec4 对齐避免对齐问题）：
 *    0-63:   viewProj (mat4x4)
 *    64-127:  modelMatrix (mat4x4)
 *    128-191: normalMatrix (mat4x4)
 *    192-207: cameraPos (vec4)
 *    208-223: lightDir (vec4)
 *    224-239: colorOpacity (vec4: xyz=颜色, w=不透明度)
 *    240-255: displayOptions (vec4: x=是否显示厚度颜色)
 */

// ─── Uniform 结构体（全部使用 vec4 对齐）───
struct Uniforms {
  viewProj: mat4x4<f32>,      // 视图投影矩阵（偏移 0-63）
  modelMatrix: mat4x4<f32>,   // 模型矩阵（偏移 64-127）
  normalMatrix: mat4x4<f32>,  // 法线矩阵（偏移 128-191）
  cameraPos: vec4<f32>,       // 相机位置（偏移 192-207）
  lightDir: vec4<f32>,        // 光照方向（偏移 208-223）
  colorOpacity: vec4<f32>,    // 颜色+不透明度（偏移 224-239）
  displayOptions: vec4<f32>,  // 显示选项（偏移 240-255）
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ─── 顶点着色器输出 ───
struct VertexOutput {
  @builtin(position) position: vec4<f32>,  // 裁剪空间坐标
  @location(0) normal: vec3<f32>,          // 世界空间法线
  @location(1) worldPos: vec3<f32>,        // 世界空间位置
  @location(2) thickness: f32,
  @location(3) localPos: vec3<f32>,
};

/**
 * 顶点着色器
 * 将顶点从模型空间变换到裁剪空间，同时计算世界空间法线
 */
@vertex
fn vsMain(
  @location(0) pos: vec3<f32>,     // 模型空间位置
  @location(1) normal: vec3<f32>,  // 模型空间法线
  @location(2) thickness: f32,
) -> VertexOutput {
  // 模型空间 → 世界空间
  let worldPos = (uniforms.modelMatrix * vec4<f32>(pos, 1.0)).xyz;
  // 法线变换（使用法线矩阵，处理非均匀缩放）
  let worldNormal = normalize((uniforms.normalMatrix * vec4<f32>(normal, 0.0)).xyz);

  var output: VertexOutput;
  output.position = uniforms.viewProj * vec4<f32>(worldPos, 1.0);
  output.normal = worldNormal;
  output.worldPos = worldPos;
  output.thickness = thickness;
  output.localPos = pos;
  return output;
}

/**
 * 片元着色器
 * 纯漫反射哑光着色
 */
fn thicknessColor(value: f32) -> vec3<f32> {
  let t = clamp(value, 0.0, 1.0);
  if (t < 0.33) {
    return mix(vec3<f32>(0.08, 0.32, 0.95), vec3<f32>(0.0, 0.9, 0.85), t / 0.33);
  }
  if (t < 0.66) {
    return mix(vec3<f32>(0.0, 0.9, 0.85), vec3<f32>(1.0, 0.88, 0.1), (t - 0.33) / 0.33);
  }
  return mix(vec3<f32>(1.0, 0.88, 0.1), vec3<f32>(0.95, 0.08, 0.04), (t - 0.66) / 0.34);
}

fn surfaceGrain(position: vec3<f32>) -> f32 {
  // 量化模型空间坐标，让磨砂颗粒固定在表面而不会随相机滑动。
  let cell = floor(position * 120.0);
  return fract(sin(dot(cell, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

@fragment
fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let lightDir = normalize(uniforms.lightDir.xyz);
  let deposited = input.thickness;
  let colorThickness = deposited * uniforms.displayOptions.y;
  let hasThickness = deposited > 0.000001;
  let thicknessOverlay = uniforms.displayOptions.z > 0.5;
  if (thicknessOverlay && !hasThickness) { discard; }
  let showThickness = hasThickness && (uniforms.displayOptions.x > 0.5 || thicknessOverlay);

  // 所有表面统一使用 Lambert 漫反射，不包含镜面高光或边缘发光。
  let diffuse = max(dot(input.normal, lightDir), 0.0);
  let ambient = 0.32;
  let baseColor = select(uniforms.colorOpacity.xyz, thicknessColor(colorThickness), showThickness);
  let grain = mix(0.965, 1.035, surfaceGrain(input.localPos));
  let color = baseColor * (ambient + diffuse * 0.55) * grain;

  // When the thickness view is enabled, keep zero-thickness faces opaque as
  // well. Otherwise the translucent mold reveals the particle bed beneath a
  // large planar face as a perfectly circular false "hole".
  // Keep the heatmap readable without making the mold an opaque screen over
  // the particle bed while deposition is still being calculated.
  let thicknessOpacity = max(uniforms.colorOpacity.w, 0.68);
  let opacity = select(uniforms.colorOpacity.w, thicknessOpacity,
    thicknessOverlay || (uniforms.displayOptions.x > 0.5 && hasThickness));
  return vec4<f32>(color, opacity);
}
