/**
 * ============================================================
 *  STEP 文件加载器 (stepLoader.js)
 * ============================================================
 *
 *  使用 OpenCascade.js（WASM 版本）解析工业标准 STEP/STP CAD 文件
 *
 *  加载方式：
 *    opencascade.full.js 是 Emscripten IIFE（非 ES Module），
 *    通过在 index.html 中用 <script> 标签加载，注册为全局变量
 *    `var Module`（工厂函数）。详见 index.html。
 *
 *  STEP 文件解析流程：
 *    1. 写入虚拟文件系统
 *    2. STEPControl_Reader 读取
 *    3. BRepMesh_IncrementalMesh 三角化
 *    4. TopExp_Explorer 遍历所有面，提取三角形
 *    5. 归一化到 [-1, 1]³ 范围
 *    6. 计算法线
 */

import initializeMMG from '@loumalouomega/mmg-wasm';
import mmgWasmUrl from '@loumalouomega/mmg-wasm/mmg-core.wasm?url';

// ─── 全局工厂函数（来自 index.html 中的 <script> 标签）───
// opencascade.full.js 用 var Module = ... 注册到全局作用域
// 在 ES Module 中通过 window.Module 访问
const OCCT_FACTORY = window.Module;
// Resolve relative to the built module so Electron file:// packaging works.
const OCCT_WASM_URL = new URL('../opencascade.full.wasm?v=20260610', import.meta.url).href;

// ─── 全局缓存（单例模式）───
let ocInstance = null;
let ocPromise = null;
let mmgPromise = null;
const UNIFORM_EDGE_LENGTH = 0.020;

/**
 * OCCT 加载状态（供外部检查加载进度和错误）
 *
 * 遵循 React 风格的 loading/error 状态管理：
 *   - loading: true 时正在加载
 *   - loaded: true 时加载完成
 *   - error: 加载失败时的错误信息
 */
export const occtState = {
  loaded: false,
  loading: false,
  error: null,
};

/**
 * 加载并初始化 OpenCascade WASM 模块
 *
 * 使用单例模式：
 *   - 已初始化：直接返回缓存实例
 *   - 正在初始化：返回进行中的 Promise
 *   - 未初始化：开始加载流程
 *
 * @param {function} [statusCallback] - 状态回调（用于 UI 显示加载进度）
 * @returns {Promise<Object>} OpenCascade 实例
 */
export function initOpenCascade(statusCallback) {
  if (ocInstance) return Promise.resolve(ocInstance);
  if (ocPromise) return ocPromise;

  ocPromise = _loadOCC(statusCallback);
  return ocPromise;
}

/**
 * 加载 OpenCascade（通过全局 Module 工厂函数）
 *
 * opencascade.full.js 已通过 index.html 的 <script> 标签加载，
 * 其 IIFE 将工厂函数赋给全局 var Module。
 * 调用工厂函数即可触发 WASM 编译 + 初始化，返回 Module.ready Promise。
 */
async function _loadOCC(statusCallback) {
  occtState.loading = true;
  occtState.error = null;

  // 检查全局工厂函数是否存在
  if (typeof OCCT_FACTORY !== 'function') {
    const errMsg = 'OpenCascade 工厂函数未找到。请确保 index.html 中已加载 <script src="/opencascade.full.js">。';
    occtState.loading = false;
    occtState.error = errMsg;
    throw new Error(errMsg);
  }

  try {
    if (statusCallback) statusCallback('正在加载 OpenCascade WASM (~30MB)...');

    // 调用工厂函数，触发 WASM 加载与初始化
    const oc = await OCCT_FACTORY({
      locateFile: (name) => {
        if (name.endsWith('.wasm')) return OCCT_WASM_URL;
        return name;
      },
    });

    // 验证 STEP API 是否就绪
    if (!oc || typeof oc.STEPControl_Reader_1 !== 'function') {
      throw new Error('OpenCascade WASM 初始化失败：STEP API 未就绪');
    }

    ocInstance = oc;
    occtState.loaded = true;
    occtState.loading = false;
    if (statusCallback) statusCallback('OpenCascade 加载成功');
    return oc;
  } catch (e) {
    occtState.loading = false;
    occtState.error = e.message;
    throw e;
  }
}

/**
 * 加载 STEP 文件并返回三角化网格
 *
 * @param {ArrayBuffer} fileData - STEP 文件的二进制内容
 * @param {number} linearDeflection - 三角化精度（相对于模型最大尺寸的比例，默认 0.05）
 * @param {function} [statusCallback] - 状态回调
 * @returns {Promise<{vertices: Float32Array, indices: Uint32Array}>}
 *   vertices: 交错格式 [x,y,z, nx,ny,nz, ...] 每顶点 6 floats
 *   indices: 三角形索引（uint32）
 *
 * 处理流程：
 *   1. 将文件写入 OpenCascade 的虚拟文件系统
 *   2. 用 STEPControl_Reader 解析 STEP 几何
 *   3. 计算包围盒，归一化到 [-0.8, 0.8] 范围
 *   4. BRepMesh 三角化（精度由 linearDeflection 控制）
 *   5. TopExp_Explorer 遍历所有面，提取三角形
 *   6. 计算平滑法线
 */
export async function loadSTEPFile(
  fileData,
  linearDeflection = 0.05,
  statusCallback,
  targetEdgeLength = UNIFORM_EDGE_LENGTH,
) {
  const oc = await initOpenCascade(statusCallback);
  const uniformEdgeLength = Math.max(0.001, Math.min(0.05, targetEdgeLength));

  if (statusCallback) statusCallback('正在解析 STEP 文件...');

  // ─── 写入虚拟文件系统 ───
  const fileName = 'input.step';
  oc.FS.writeFile('/' + fileName, new Uint8Array(fileData));

  // ─── 解析 STEP 文件 ───
  const reader = new oc.STEPControl_Reader_1();
  const status = reader.ReadFile(fileName);

  // ReadFile 返回 embind 枚举对象（ctor {value: 1}），无法用 Number() 或 === 比较
  // 改用 NbRootsForTransfer() 验证文件是否成功读取
  const nRoots = reader.NbRootsForTransfer();
  if (nRoots === 0) {
    console.warn('ReadFile status:', typeof status, status, 'NbRootsForTransfer:', nRoots);
    throw new Error('STEP 文件读取失败。请检查文件格式。');
  }

  // 传输所有根实体（需要传递一个 ProgressRange 对象）
  reader.TransferRoots(new oc.Message_ProgressRange_1());
  const shape = reader.OneShape();  // 获取合并后的形状

  // ─── 计算包围盒 ───
  const bbox = new oc.Bnd_Box_1();
  oc.BRepBndLib.Add(shape, bbox, false);
  const minP = bbox.CornerMin();
  const maxP = bbox.CornerMax();
  const xMin = minP.X(), yMin = minP.Y(), zMin = minP.Z();
  const xMax = maxP.X(), yMax = maxP.Y(), zMax = maxP.Z();
  const sizeX = xMax - xMin, sizeY = yMax - yMin, sizeZ = zMax - zMin;
  const maxSize = Math.max(sizeX, sizeY, sizeZ);
  const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2, cz = (zMin + zMax) / 2;
  const scale = 1.6 / maxSize;  // 归一化到 [-0.8, 0.8]（留出边距）

  if (statusCallback) statusCallback('正在三角化网格...');

  // ─── 三角化 ───
  // linearDeflection * maxSize = 最大允许的线性偏差（绝对值）
  // 值越小，三角形越密，精度越高
  new oc.BRepMesh_IncrementalMesh_2(shape, linearDeflection * maxSize, false, 0.15, true);

  // ─── 提取三角形 ───
  const vertices = [], indices = [];
  let vertexOffset = 0;

  // 用 TopExp_Explorer 遍历形状中的所有面
  const explorer = new oc.TopExp_Explorer_2(
    shape, oc.TopAbs_ShapeEnum.TopAbs_FACE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE
  );

  while (explorer.More()) {
    const face = oc.TopoDS.Face_1(explorer.Current());
    const location = new oc.TopLoc_Location_1();
    const triHandle = oc.BRep_Tool.Triangulation(face, location, 0);

    if (!triHandle.IsNull()) {
      const tri = triHandle.get();  // Handle → Poly_Triangulation
      const transform = location.Transformation();
      const nbNodes = tri.NbNodes();       // 顶点数
      const nbTriangles = tri.NbTriangles(); // 三角形数

      // 提取顶点（变换到世界坐标，归一化到 [-0.8, 0.8]）
      for (let i = 1; i <= nbNodes; i++) {
        const node = tri.Node(i);  // 返回 gp_Pnt
        node.Transform(transform);  // 原地应用面的局部变换
        vertices.push(
          (node.X() - cx) * scale, (node.Y() - cy) * scale, (node.Z() - cz) * scale,
          0, 0, 1  // 法线占位（稍后计算）
        );
      }

      // 提取三角形索引（注意 OpenCascade 索引从 1 开始，需要 -1）
      // 同时翻转 v1/v2 顺序修正面法线方向
      for (let i = 1; i <= nbTriangles; i++) {
        const triIdx = tri.Triangle(i);
        indices.push(
          triIdx.Value(1) - 1 + vertexOffset,
          triIdx.Value(3) - 1 + vertexOffset,  // 翻转顺序
          triIdx.Value(2) - 1 + vertexOffset
        );
      }

      vertexOffset += nbNodes;
    }
    explorer.Next();
  }

  // 清理虚拟文件系统
  try { oc.FS.unlink('/' + fileName); } catch (_) {}

  if (vertices.length === 0) {
    throw new Error('无法从 STEP 文件中提取三角形网格。');
  }

  // OpenCascade 负责提供可靠的 CAD 曲面近似，MMGS 再按常量尺寸场重生成
  // 有限元级各向同性三角网格。hmin/hmax 被锁在目标尺寸的 ±5%。
  if (statusCallback) statusCallback('正在生成全局均匀有限元网格...');
  const refined = await remeshSurfaceWithMMG(vertices, indices, uniformEdgeLength);
  const shaded = splitVerticesAtCreases(refined.vertices, refined.indices, Math.PI / 4);
  computeNormalsInPlace(shaded.vertices, shaded.indices);

  return {
    vertices: new Float32Array(shaded.vertices),
    indices: new Uint32Array(shaded.indices),
    statistics: refined.statistics || meshEdgeStatistics(shaded.vertices, shaded.indices, uniformEdgeLength),
    analysisVertices: new Float32Array(vertices),
    analysisIndices: new Uint32Array(indices),
  };
}

export async function remeshSurfaceWithMMG(sourceVertices, sourceIndices, targetLength) {
  if (!mmgPromise) {
    mmgPromise = initializeMMG({
      locateFile: (path) => path.endsWith('.wasm') ? mmgWasmUrl : path,
    });
  }
  const { mmgs } = await mmgPromise;
  const welded = weldCoincidentVertices(sourceVertices, sourceIndices, 1e-6);
  const vertexCount = welded.vertices.length / 6;
  const triangleCount = welded.indices.length / 3;
  const coordinates = new Float64Array(vertexCount * 3);
  const triangles = new Int32Array(welded.indices.length);

  for (let i = 0; i < vertexCount; i++) {
    coordinates[i * 3] = welded.vertices[i * 6];
    coordinates[i * 3 + 1] = welded.vertices[i * 6 + 1];
    coordinates[i * 3 + 2] = welded.vertices[i * 6 + 2];
  }
  for (let i = 0; i < welded.indices.length; i++) triangles[i] = welded.indices[i] + 1;

  const handles = mmgs.init();
  try {
    mmgs.setMeshSize(handles.mesh, vertexCount, triangleCount, 0);
    mmgs.setVertices(handles.mesh, coordinates, null);
    mmgs.setTriangles(handles.mesh, triangles, null);

    mmgs.setIparameter(handles.mesh, handles.met, mmgs.IPARAM_verbose, -1);
    mmgs.setIparameter(handles.mesh, handles.met, mmgs.IPARAM_angle, 1);
    // 不让 STEP 特征线上原有的短边继续向尺寸场传播，否则少数 CAD 接缝
    // 会形成明显的局部高密度带；锐边本身仍由 angle 检测保留。
    mmgs.setIparameter(handles.mesh, handles.met, mmgs.IPARAM_nosizreq, 1);
    mmgs.setDparameter(handles.mesh, handles.met, mmgs.DPARAM_angleDetection, 40);
    mmgs.setDparameter(handles.mesh, handles.met, mmgs.DPARAM_hsiz, targetLength);
    mmgs.setDparameter(handles.mesh, handles.met, mmgs.DPARAM_hmin, targetLength * 0.95);
    mmgs.setDparameter(handles.mesh, handles.met, mmgs.DPARAM_hmax, targetLength * 1.05);
    mmgs.setDparameter(handles.mesh, handles.met, mmgs.DPARAM_hausd, targetLength * 0.3);
    mmgs.setDparameter(handles.mesh, handles.met, mmgs.DPARAM_hgrad, 1.001);
    mmgs.remesh(handles.mesh, handles.met);

    const size = mmgs.getMeshSize(handles.mesh);
    if (size.np === 0 || size.nt === 0) throw new Error('MMGS returned an empty mesh');
    const outputVertices = mmgs.getVertices(handles.mesh, size.np).vertices;
    const outputTriangles = mmgs.getTriangles(handles.mesh, size.nt).tria;
    const vertices = new Array(size.np * 6);
    const indices = new Array(size.nt * 3);

    for (let i = 0; i < size.np; i++) {
      vertices[i * 6] = outputVertices[i * 3];
      vertices[i * 6 + 1] = outputVertices[i * 3 + 1];
      vertices[i * 6 + 2] = outputVertices[i * 3 + 2];
      vertices[i * 6 + 3] = 0;
      vertices[i * 6 + 4] = 0;
      vertices[i * 6 + 5] = 0;
    }
    for (let i = 0; i < outputTriangles.length; i++) indices[i] = outputTriangles[i] - 1;

    const stats = meshEdgeStatistics(vertices, indices, targetLength);
    console.info('[MMGS] Uniform mesh:', { vertices: size.np, triangles: size.nt, ...stats });
    if (stats.p95 > targetLength * 1.45 || stats.p05 < targetLength * 0.55) {
      console.warn('[MMGS] Geometry constraints forced some edges outside the requested size band', stats);
    }
    return { vertices, indices, statistics: stats };
  } finally {
    mmgs.free(handles);
  }
}

function meshEdgeStatistics(vertices, indices, targetLength) {
  const lengths = [];
  const unique = new Set();
  for (let i = 0; i < indices.length; i += 3) {
    for (const [a, b] of [[indices[i], indices[i + 1]], [indices[i + 1], indices[i + 2]], [indices[i + 2], indices[i]]]) {
      const key = edgeKey(a, b);
      if (unique.has(key)) continue;
      unique.add(key);
      lengths.push(Math.sqrt(vertexDistanceSq(vertices, a, b)));
    }
  }
  lengths.sort((a, b) => a - b);
  const mean = lengths.reduce((sum, value) => sum + value, 0) / Math.max(1, lengths.length);
  const deviation = Math.sqrt(lengths.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, lengths.length));
  return {
    target: targetLength,
    min: lengths[0] || 0,
    p05: lengths[Math.floor(lengths.length * 0.05)] || 0,
    mean,
    p95: lengths[Math.floor(lengths.length * 0.95)] || 0,
    max: lengths[lengths.length - 1] || 0,
    coefficientOfVariation: mean > 0 ? deviation / mean : 0,
  };
}

/**
 * 对所有超长共享边同步取中点，避免相邻三角形产生 T 形裂缝。
 * 该步骤控制的是单元尺寸；CAD 曲面精度仍由 OpenCascade 三角化负责。
 */
function remeshIsotropic(sourceVertices, sourceIndices, targetLength) {
  // STEP 中相邻 CAD 面通常各自保存一份公共边顶点。先焊接，否则这些边会被
  // 误判为开放边界，短边无法合并，形成沿 CAD 面边界的局部高密度带。
  const welded = weldCoincidentVertices(sourceVertices, sourceIndices, 1e-6);
  const vertices = welded.vertices;
  let indices = welded.indices;
  const splitSq = (targetLength * 4 / 3) ** 2;
  const maxTriangles = 250000;

  const edgeLengthSq = (a, b) => {
    const ai = a * 6, bi = b * 6;
    const dx = vertices[ai] - vertices[bi];
    const dy = vertices[ai + 1] - vertices[bi + 1];
    const dz = vertices[ai + 2] - vertices[bi + 2];
    return dx * dx + dy * dy + dz * dz;
  };

  for (let pass = 0; pass < 12; pass++) {
    const splitEdges = new Set();
    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t], b = indices[t + 1], c = indices[t + 2];
      for (const [p, q] of [[a, b], [b, c], [c, a]]) {
        if (edgeLengthSq(p, q) > splitSq) {
          splitEdges.add(edgeKey(p, q));
        }
      }
    }
    if (splitEdges.size === 0) break;

    const midpointIndices = new Map();
    const midpoint = (a, b) => {
      const key = edgeKey(a, b);
      let result = midpointIndices.get(key);
      if (result !== undefined) return result;
      const ai = a * 6, bi = b * 6;
      result = vertices.length / 6;
      vertices.push(
        (vertices[ai] + vertices[bi]) * 0.5,
        (vertices[ai + 1] + vertices[bi + 1]) * 0.5,
        (vertices[ai + 2] + vertices[bi + 2]) * 0.5,
        0, 0, 0,
      );
      midpointIndices.set(key, result);
      return result;
    };

    const next = [];
    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t], b = indices[t + 1], c = indices[t + 2];
      const splitAB = splitEdges.has(edgeKey(a, b));
      const splitBC = splitEdges.has(edgeKey(b, c));
      const splitCA = splitEdges.has(edgeKey(c, a));
      const count = Number(splitAB) + Number(splitBC) + Number(splitCA);

      if (count === 0) {
        next.push(a, b, c);
      } else if (count === 1 && splitAB) {
        const ab = midpoint(a, b);
        next.push(a, ab, c, ab, b, c);
      } else if (count === 1 && splitBC) {
        const bc = midpoint(b, c);
        next.push(b, bc, a, bc, c, a);
      } else if (count === 1) {
        const ca = midpoint(c, a);
        next.push(c, ca, b, ca, a, b);
      } else if (count === 2 && !splitCA) {
        const ab = midpoint(a, b), bc = midpoint(b, c);
        next.push(b, bc, ab, a, ab, c, ab, bc, c);
      } else if (count === 2 && !splitBC) {
        const ab = midpoint(a, b), ca = midpoint(c, a);
        next.push(a, ab, ca, ab, b, c, ab, c, ca);
      } else if (count === 2) {
        const bc = midpoint(b, c), ca = midpoint(c, a);
        next.push(c, ca, bc, b, bc, a, bc, ca, a);
      } else {
        const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
        next.push(a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca);
      }
    }
    indices = next;
    if (indices.length / 3 > maxTriangles) {
      console.warn(`Uniform refinement stopped at ${indices.length / 3} triangles`);
      break;
    }
  }
  const postProcessMaxEdge = targetLength * 4 / 3;
  indices = collapseShortEdges(vertices, indices, targetLength * 0.78, postProcessMaxEdge);
  indices = improveTriangleQuality(vertices, indices, 8, postProcessMaxEdge);
  return compactMesh(vertices, indices);
}

function weldCoincidentVertices(sourceVertices, sourceIndices, tolerance) {
  const vertices = [];
  const remap = new Uint32Array(sourceVertices.length / 6);
  const buckets = new Map();

  for (let oldIndex = 0; oldIndex < remap.length; oldIndex++) {
    const i = oldIndex * 6;
    const key = [sourceVertices[i], sourceVertices[i + 1], sourceVertices[i + 2]]
      .map(value => Math.round(value / tolerance)).join(',');
    let newIndex = buckets.get(key);
    if (newIndex === undefined) {
      newIndex = vertices.length / 6;
      buckets.set(key, newIndex);
      vertices.push(sourceVertices[i], sourceVertices[i + 1], sourceVertices[i + 2], 0, 0, 0);
    }
    remap[oldIndex] = newIndex;
  }

  const indices = [];
  for (let i = 0; i < sourceIndices.length; i += 3) {
    const a = remap[sourceIndices[i]], b = remap[sourceIndices[i + 1]], c = remap[sourceIndices[i + 2]];
    if (a !== b && b !== c && c !== a) indices.push(a, b, c);
  }
  return { vertices, indices };
}

/** 拆分锐边两侧的共享顶点，同时保留同一光滑曲面上的法线连续性。 */
function splitVerticesAtCreases(sourceVertices, sourceIndices, creaseAngle) {
  const triangleCount = sourceIndices.length / 3;
  const faceNormals = new Array(triangleCount);
  const incident = new Array(sourceVertices.length / 6).fill(null).map(() => []);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const i = triangle * 3;
    const a = sourceIndices[i], b = sourceIndices[i + 1], c = sourceIndices[i + 2];
    faceNormals[triangle] = triangleNormal(sourceVertices, a, b, c);
    incident[a].push(triangle);
    incident[b].push(triangle);
    incident[c].push(triangle);
  }

  const cosineThreshold = Math.cos(creaseAngle);
  const vertices = [];
  const cornerMap = new Map();

  for (let vertex = 0; vertex < incident.length; vertex++) {
    const faces = incident[vertex];
    if (faces.length === 0) continue;
    const parent = faces.map((_, index) => index);
    const find = index => {
      while (parent[index] !== index) {
        parent[index] = parent[parent[index]];
        index = parent[index];
      }
      return index;
    };
    const union = (a, b) => {
      const ra = find(a), rb = find(b);
      if (ra !== rb) parent[rb] = ra;
    };
    for (let a = 0; a < faces.length; a++) for (let b = a + 1; b < faces.length; b++) {
      if (dot3(faceNormals[faces[a]], faceNormals[faces[b]]) >= cosineThreshold) union(a, b);
    }

    const groupVertices = new Map();
    for (let faceIndex = 0; faceIndex < faces.length; faceIndex++) {
      const root = find(faceIndex);
      let newVertex = groupVertices.get(root);
      if (newVertex === undefined) {
        newVertex = vertices.length / 6;
        groupVertices.set(root, newVertex);
        const i = vertex * 6;
        vertices.push(sourceVertices[i], sourceVertices[i + 1], sourceVertices[i + 2], 0, 0, 0);
      }
      cornerMap.set(`${faces[faceIndex]},${vertex}`, newVertex);
    }
  }

  const indices = [];
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const i = triangle * 3;
    indices.push(
      cornerMap.get(`${triangle},${sourceIndices[i]}`),
      cornerMap.get(`${triangle},${sourceIndices[i + 1]}`),
      cornerMap.get(`${triangle},${sourceIndices[i + 2]}`),
    );
  }
  return { vertices, indices };
}

function edgeKey(a, b) {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function collapseShortEdges(vertices, sourceIndices, minLength, maxLength) {
  let indices = sourceIndices;
  const minSq = minLength * minLength;
  const maxSq = maxLength * maxLength;

  for (let pass = 0; pass < 6; pass++) {
    const edgeMap = buildEdgeMap(indices);
    const boundary = new Set();
    const neighbors = new Map();
    const incidentTriangles = buildIncidentTriangles(indices);
    const candidates = [];

    for (const records of edgeMap.values()) {
      const { a, b } = records[0];
      if (records.length === 1) {
        boundary.add(a);
        boundary.add(b);
      }
      addNeighbor(neighbors, a, b);
      addNeighbor(neighbors, b, a);
      const lengthSq = vertexDistanceSq(vertices, a, b);
      if (records.length === 2 && lengthSq < minSq) candidates.push({ a, b, lengthSq });
    }
    candidates.sort((x, y) => x.lengthSq - y.lengthSq);

    const replacement = new Map();
    const used = new Set();
    let collapsed = 0;
    for (const { a, b } of candidates) {
      if (used.has(a) || used.has(b) || boundary.has(a) || boundary.has(b)) continue;
      const common = countCommon(neighbors.get(a), neighbors.get(b));
      if (!collapseKeepsEdgeLengths(vertices, neighbors, a, b, maxSq)) continue;
      if (common !== 2 || !collapsePreservesFaces(vertices, indices, incidentTriangles, a, b)) continue;

      const ai = a * 6, bi = b * 6;
      vertices[ai] = (vertices[ai] + vertices[bi]) * 0.5;
      vertices[ai + 1] = (vertices[ai + 1] + vertices[bi + 1]) * 0.5;
      vertices[ai + 2] = (vertices[ai + 2] + vertices[bi + 2]) * 0.5;
      replacement.set(b, a);
      used.add(a);
      used.add(b);
      // 相邻边在同一轮同时折叠会使两端都移动，破坏最大边长约束。
      for (const vertex of neighbors.get(a) || []) used.add(vertex);
      for (const vertex of neighbors.get(b) || []) used.add(vertex);
      collapsed++;
    }
    if (collapsed === 0) break;

    const next = [];
    for (let i = 0; i < indices.length; i += 3) {
      const a = replacement.get(indices[i]) ?? indices[i];
      const b = replacement.get(indices[i + 1]) ?? indices[i + 1];
      const c = replacement.get(indices[i + 2]) ?? indices[i + 2];
      if (a !== b && b !== c && c !== a) next.push(a, b, c);
    }
    indices = next;
  }
  return indices;
}

function improveTriangleQuality(vertices, sourceIndices, maxPasses, maxEdgeLength) {
  let indices = sourceIndices.slice();
  const maxEdgeSq = maxEdgeLength * maxEdgeLength;
  for (let pass = 0; pass < maxPasses; pass++) {
    const edgeMap = buildEdgeMap(indices);
    const usedTriangles = new Set();
    let flipped = 0;

    for (const records of edgeMap.values()) {
      if (records.length !== 2) continue;
      const first = records[0], second = records[1];
      if (usedTriangles.has(first.triangle) || usedTriangles.has(second.triangle)) continue;
      if (first.a !== second.b || first.b !== second.a) continue;
      const a = first.a, b = first.b, c = first.opposite, d = second.opposite;
      if (c === d || edgeMap.has(edgeKey(c, d))) continue;
      if (vertexDistanceSq(vertices, c, d) > maxEdgeSq) continue;

      const before = Math.min(triangleQuality(vertices, a, b, c), triangleQuality(vertices, b, a, d));
      const after = Math.min(triangleQuality(vertices, c, d, b), triangleQuality(vertices, d, c, a));
      if (after <= before + 0.04 || !flipPreservesSurface(vertices, a, b, c, d)) continue;

      indices.splice(first.triangle * 3, 3, c, d, b);
      indices.splice(second.triangle * 3, 3, d, c, a);
      usedTriangles.add(first.triangle);
      usedTriangles.add(second.triangle);
      flipped++;
    }
    if (flipped === 0) break;
  }
  return indices;
}

function buildEdgeMap(indices) {
  const map = new Map();
  for (let i = 0; i < indices.length; i += 3) {
    const triangle = i / 3;
    const a = indices[i], b = indices[i + 1], c = indices[i + 2];
    for (const [p, q, opposite] of [[a, b, c], [b, c, a], [c, a, b]]) {
      const key = edgeKey(p, q);
      if (!map.has(key)) map.set(key, []);
      map.get(key).push({ a: p, b: q, opposite, triangle });
    }
  }
  return map;
}

function addNeighbor(neighbors, a, b) {
  if (!neighbors.has(a)) neighbors.set(a, new Set());
  neighbors.get(a).add(b);
}

function countCommon(a, b) {
  if (!a || !b) return 0;
  let count = 0;
  for (const value of a) if (b.has(value)) count++;
  return count;
}

function collapseKeepsEdgeLengths(vertices, neighbors, a, b, maxSq) {
  const ai = a * 6, bi = b * 6;
  const midpoint = [
    (vertices[ai] + vertices[bi]) * 0.5,
    (vertices[ai + 1] + vertices[bi + 1]) * 0.5,
    (vertices[ai + 2] + vertices[bi + 2]) * 0.5,
  ];
  const adjacent = new Set([...(neighbors.get(a) || []), ...(neighbors.get(b) || [])]);
  adjacent.delete(a);
  adjacent.delete(b);
  for (const vertex of adjacent) {
    if (distanceSq3(midpoint, vertexPosition(vertices, vertex)) > maxSq) return false;
  }
  return true;
}

function buildIncidentTriangles(indices) {
  const incident = new Map();
  for (let i = 0; i < indices.length; i += 3) {
    const triangle = i / 3;
    for (const vertex of [indices[i], indices[i + 1], indices[i + 2]]) {
      if (!incident.has(vertex)) incident.set(vertex, new Set());
      incident.get(vertex).add(triangle);
    }
  }
  return incident;
}

function collapsePreservesFaces(vertices, indices, incidentTriangles, keep, remove) {
  const ki = keep * 6, ri = remove * 6;
  const midpoint = [
    (vertices[ki] + vertices[ri]) * 0.5,
    (vertices[ki + 1] + vertices[ri + 1]) * 0.5,
    (vertices[ki + 2] + vertices[ri + 2]) * 0.5,
  ];
  const affected = new Set([...(incidentTriangles.get(keep) || []), ...(incidentTriangles.get(remove) || [])]);
  for (const triangle of affected) {
    const i = triangle * 3;
    const tri = [indices[i], indices[i + 1], indices[i + 2]];
    if (!tri.includes(keep) && !tri.includes(remove)) continue;
    const mapped = tri.map(v => v === remove ? keep : v);
    if (new Set(mapped).size < 3) continue;
    const oldNormal = triangleNormal(vertices, tri[0], tri[1], tri[2]);
    const points = mapped.map(v => v === keep ? midpoint : vertexPosition(vertices, v));
    const newNormal = normalFromPoints(points[0], points[1], points[2]);
    if (dot3(oldNormal, newNormal) < 0.65) return false;
  }
  return true;
}

function flipPreservesSurface(vertices, a, b, c, d) {
  const n1 = triangleNormal(vertices, a, b, c);
  const n2 = triangleNormal(vertices, b, a, d);
  const n3 = triangleNormal(vertices, c, d, b);
  const n4 = triangleNormal(vertices, d, c, a);
  return dot3(n1, n3) > 0.65 && dot3(n2, n4) > 0.65;
}

function triangleQuality(vertices, a, b, c) {
  const pa = vertexPosition(vertices, a), pb = vertexPosition(vertices, b), pc = vertexPosition(vertices, c);
  const ab = distanceSq3(pa, pb), bc = distanceSq3(pb, pc), ca = distanceSq3(pc, pa);
  const n = normalFromPoints(pa, pb, pc, false);
  const twiceArea = Math.sqrt(dot3(n, n));
  return (2 * Math.sqrt(3) * twiceArea) / Math.max(1e-20, ab + bc + ca);
}

function triangleNormal(vertices, a, b, c) {
  return normalFromPoints(vertexPosition(vertices, a), vertexPosition(vertices, b), vertexPosition(vertices, c));
}

function normalFromPoints(a, b, c, normalize = true) {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
  const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
  const n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
  if (!normalize) return n;
  const length = Math.sqrt(dot3(n, n));
  return length > 1e-14 ? n.map(value => value / length) : [0, 0, 0];
}

function vertexPosition(vertices, index) {
  const i = index * 6;
  return [vertices[i], vertices[i + 1], vertices[i + 2]];
}

function vertexDistanceSq(vertices, a, b) {
  return distanceSq3(vertexPosition(vertices, a), vertexPosition(vertices, b));
}

function distanceSq3(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function compactMesh(vertices, indices) {
  const remap = new Map();
  const compactVertices = [];
  const compactIndices = [];
  for (const oldIndex of indices) {
    if (!remap.has(oldIndex)) {
      const newIndex = compactVertices.length / 6;
      remap.set(oldIndex, newIndex);
      const i = oldIndex * 6;
      compactVertices.push(vertices[i], vertices[i + 1], vertices[i + 2], 0, 0, 0);
    }
    compactIndices.push(remap.get(oldIndex));
  }
  return { vertices: compactVertices, indices: compactIndices };
}

/**
 * 就地计算平滑法线
 *
 * 算法：
 *   1. 初始化所有顶点法线为零
 *   2. 对每个三角形，计算面法线（叉积）
 *   3. 将面法线累加到三个顶点（面积加权）
 *   4. 归一化每个顶点的法线
 *
 * 这种方法实现了面积加权的平滑法线：
 * 大三角形对面法线的贡献更大，小三角形贡献更小
 *
 * @param {number[]} vertices - 交错顶点数组 [x,y,z, nx,ny,nz, ...]
 * @param {Uint32Array} indices - 三角形索引
 */
function computeNormalsInPlace(vertices, indices) {
  const numVerts = vertices.length / 6;

  // ─── 步骤 1: 清零所有法线 ───
  for (let i = 0; i < numVerts; i++) {
    vertices[i * 6 + 3] = 0;
    vertices[i * 6 + 4] = 0;
    vertices[i * 6 + 5] = 0;
  }

  // ─── 步骤 2: 累加面法线到顶点 ───
  const numTris = indices.length / 3;
  for (let t = 0; t < numTris; t++) {
    const i0 = indices[t * 3] * 6;
    const i1 = indices[t * 3 + 1] * 6;
    const i2 = indices[t * 3 + 2] * 6;

    // 三个顶点位置
    const ax = vertices[i0], ay = vertices[i0 + 1], az = vertices[i0 + 2];
    const bx = vertices[i1], by = vertices[i1 + 1], bz = vertices[i1 + 2];
    const cx = vertices[i2], cy = vertices[i2 + 1], cz = vertices[i2 + 2];

    // 两个边向量
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;

    // 叉积 → 面法线（面积加权）
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;

    // 累加到三个顶点
    vertices[i0 + 3] += nx;
    vertices[i0 + 4] += ny;
    vertices[i0 + 5] += nz;
    vertices[i1 + 3] += nx;
    vertices[i1 + 4] += ny;
    vertices[i1 + 5] += nz;
    vertices[i2 + 3] += nx;
    vertices[i2 + 4] += ny;
    vertices[i2 + 5] += nz;
  }

  // ─── 步骤 3: 归一化法线 ───
  for (let i = 0; i < numVerts; i++) {
    const idx = i * 6;
    const nx = vertices[idx + 3];
    const ny = vertices[idx + 4];
    const nz = vertices[idx + 5];
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-10) {
      vertices[idx + 3] = nx / len;
      vertices[idx + 4] = ny / len;
      vertices[idx + 5] = nz / len;
    }
  }
}

/**
 * 使用 OpenCascade 内核生成立方体三角网格
 *
 * 通过 BRepPrimAPI_MakeBox 创建精确的立方体实体，
 * 再用 BRepMesh 三角化，得到流形且闭合的三角网格。
 *
 * @param {Object} oc - OpenCascade 实例（需先调用 initOpenCascade）
 * @param {number} halfSize - 立方体半边长（默认 0.8，总边长 1.6）
 * @param {number} deflection - 三角化最大偏差（默认 0.015）
 * @returns {{vertices: Float32Array, indices: Uint32Array}}
 *   vertices: 交错 [x,y,z, nx,ny,nz] stride=6
 *   indices: 三角形索引
 */
export function generateOCCCube(oc, halfSize = 0.8, deflection = 0.015) {
  const size = halfSize * 2;

  // 创建立方体实体
  const box = new oc.BRepPrimAPI_MakeBox_2(size, size, size);
  const shape = box.Shape();

  // 三角化
  new oc.BRepMesh_IncrementalMesh_2(shape, deflection, false, 0.5, true);

  // 提取三角形
  const vertices = [];
  const indices = [];
  let vertexOffset = 0;

  const explorer = new oc.TopExp_Explorer_2(
    shape, oc.TopAbs_ShapeEnum.TopAbs_FACE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE
  );

  while (explorer.More()) {
    const face = oc.TopoDS.Face_1(explorer.Current());
    const location = new oc.TopLoc_Location_1();
    const triHandle = oc.BRep_Tool.Triangulation(face, location, 0);

    if (!triHandle.IsNull()) {
      const tri = triHandle.get();
      const transform = location.Transformation();
      const nbNodes = tri.NbNodes();
      const nbTriangles = tri.NbTriangles();

      for (let i = 1; i <= nbNodes; i++) {
        const node = tri.Node(i);
        node.Transform(transform);
        // 平移使立方体中心在原点
        vertices.push(
          node.X() - halfSize,
          node.Y() - halfSize,
          node.Z() - halfSize,
          0, 0, 1
        );
      }

      for (let i = 1; i <= nbTriangles; i++) {
        const triIdx = tri.Triangle(i);
        indices.push(
          triIdx.Value(1) - 1 + vertexOffset,
          triIdx.Value(3) - 1 + vertexOffset,
          triIdx.Value(2) - 1 + vertexOffset
        );
      }

      vertexOffset += nbNodes;
    }
    explorer.Next();
  }

  if (vertices.length === 0) {
    throw new Error('OCCT 立方体生成失败：未提取到三角形');
  }

  const refined = remeshIsotropic(vertices, indices, 0.065);
  const shaded = splitVerticesAtCreases(refined.vertices, refined.indices, Math.PI / 4);
  computeNormalsInPlace(shaded.vertices, shaded.indices);

  return {
    vertices: new Float32Array(shaded.vertices),
    indices: new Uint32Array(shaded.indices),
  };
}

/**
 * 平面着色法线计算 — 每个三角形独立计算法线，
 * 三个顶点使用相同的面法线，没有平滑过渡。
 * 适用于立方体等有硬边（hard edge）的模型。
 */
function computeFlatNormalsInPlace(vertices, indices) {
  const numTris = indices.length / 3;
  for (let t = 0; t < numTris; t++) {
    const i0 = indices[t * 3] * 6;
    const i1 = indices[t * 3 + 1] * 6;
    const i2 = indices[t * 3 + 2] * 6;

    // 三个顶点位置
    const ax = vertices[i0], ay = vertices[i0 + 1], az = vertices[i0 + 2];
    const bx = vertices[i1], by = vertices[i1 + 1], bz = vertices[i1 + 2];
    const cx = vertices[i2], cy = vertices[i2 + 1], cz = vertices[i2 + 2];

    // 两个边向量 → 叉积 → 面法线
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-10) { nx /= len; ny /= len; nz /= len; }

    // 直接赋值（不平滑平均）
    vertices[i0 + 3] = nx; vertices[i0 + 4] = ny; vertices[i0 + 5] = nz;
    vertices[i1 + 3] = nx; vertices[i1 + 4] = ny; vertices[i1 + 5] = nz;
    vertices[i2 + 3] = nx; vertices[i2 + 4] = ny; vertices[i2 + 5] = nz;
  }
}
