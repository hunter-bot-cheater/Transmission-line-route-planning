(() => {
  const $ = (id) => document.getElementById(id);
  const SCAN_UPDATE_DISTANCE = 4.5;
  const MAX_CANVAS_PIXELS = 16 * 1024 * 1024;
  const API_PROTOCOL = 'server-authorized-scan-v4';
  const WORLD_CAPACITY_ERROR = '地图会话容量已满；请关闭其他地图标签页，或等待空闲会话过期后重试';
  const PAGE_ID = window.crypto?.randomUUID?.()||`${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  const ACTIVE_SESSION_STORAGE_KEY = 'terrain-scan.active-session-id';
  const PENDING_WORLD_STORAGE_KEY = 'terrain-scan.pending-world-create';
  const canvas = $('mapCanvas'), ctx = canvas.getContext('2d');
  const baseCanvas = document.createElement('canvas'), baseCtx = baseCanvas.getContext('2d');
    const state = { world: null, worldRequestSeed: null, worldLoading: false, worldLoadingFocus: null, worldLoadFailed: false, sessionExpired: false, pageSessionReleased: false, sessionSnapshotRequired: false, sessionSnapshotPromise: null, sessionSnapshotRevision: 0, serverSnapshotVersion: null, pendingServerSnapshotRevision: null, pendingServerSnapshotRetryRevision: null, worldSessionUnconfirmed: false, pendingWorldCreate: null, healthCheckPromise: null, serverInstance: null, apiProtocolMismatch: false, plannerReady: null, plannerError: null, plannerFailureRevision: 0, controlToken: null, controlEpoch: 0, controlGeneration: 0, controlOwner: null, controlBusy: false, controlLost: false, controlAcquirePromise: null, controlAcquireError: null, startingFlight: false, observed: [], confidence: [], samples: [], route: [], scanPath: [], scanTrail: [], flightDistance: 0, flying: false, paused: false, tick: null, tickRunningRevision: null, resumePending: false, retryScanSelection: false, endpointRevision: 0, missionSyncedRevision: 0, missionUpdateQueue: Promise.resolve(true), movementFrame: null, movement: null, dronePosition: null, previousDronePosition: null, scanSincePlan: 0, planCount: 0, guidanceWeight: 0.75, coverage: 0, announcedCoverageMilestone: 0, activePoint: 'start', keyboardPoint: null, keyboardMapUsed: false, pointerClient: null, pointerReadoutActive: false, mapZoom: 1, mapViewportWidth: 0, mapViewportHeight: 0, connected: null, lastDrone: null, costGridCache: null, layerRangeCache: {}, layerLegendSignature: '', baseSignature: '', dataRevision: 0, worldRevision: 0, planRevision: 0, plannedDataRevision: -1, planning: false, planQueued: false, retryPlanPending: false, planAfterScan: false, pendingScan: null, reconcilingScan: false };
  state.scanRequestCapacityReached=false;
  let startupReplacementSessionId = null;
  const layerLabels = { elevation:'归一化高程',slope:'坡度（归一化）',aspect_cos:'合成坡向余弦编码',aspect_sin:'合成坡向正弦编码',tri:'归一化粗糙度代理',tpi_100:'地形位置代理（中/大尺度差分）',tpi_300:'地形位置代理（细/大尺度差分）',tpi_900:'归一化高程偏移代理',profile_curvature:'剖面曲率代理（归一化）',plan_curvature:'平面曲率代理（归一化）',roughness_3:'细尺度起伏代理',roughness_9:'中尺度起伏代理',roughness_27:'大尺度起伏代理',water:'水域',dist_water:'水域反向强度（合成）',road:'道路',dist_road:'道路反向强度（合成）',railway:'铁路',dist_railway:'铁路关联场（合成）',fault:'断裂带',dist_fault:'断裂带关联场（合成）',existing_line:'既有输电线',dist_existing_line:'既有线关联场（合成）',landuse:'土地利用',building:'建筑密度',vegetation:'植被高度',protected:'保护区',seismic:'地震风险',landslide:'滑坡风险',typhoon:'台风风险',ice:'覆冰风险',lightning:'雷击风险',hard_blocked:'硬约束区',confidence:'观测置信度',cost:'综合代价' };
  const requiredWorldLayers = Object.keys(layerLabels).filter(name=>name!=='confidence'&&name!=='cost');
  const worldWeightNames = ['slope','tri','roughness_9','landuse','building','vegetation','protected','seismic','landslide','typhoon','ice','lightning','dist_water','dist_fault','dist_railway','dist_existing_line','dist_road'];
  // The simulator uses a 50m grid and weights normalized-layer costs in [0,1].
  const CELL_SIZE_METERS=50, MAX_LAYER_WEIGHT=1;
  // Keep aligned with the V5 server's per-session request receipt cap and scan spacing.
  const MAX_SCAN_RECEIPTS=8192, SCAN_CENTER_SPACING=1.5;
  const derivedProxyLayers = ['aspect_cos','aspect_sin','tri','tpi_100','tpi_300','tpi_900','profile_curvature','plan_curvature','roughness_3','roughness_9','roughness_27'];
  const terrainColors = [[24,57,45],[56,96,72],[135,147,94],[193,167,107],[143,121,88],[225,217,189]];
  const landuseColors = ['#46694d','#bcaa69','#a98758','#705e76','#367e9b','#8b684f'];
  const riskColors = [[24,76,71],[91,157,118],[215,189,101],[232,122,75],[169,52,72]];
  const clamp = (n, a=0, b=1) => Math.max(a, Math.min(b,n));
  const lerp = (a,b,t) => a + (b-a)*t;
  const interp = (stops, t) => { t=clamp(t)*(stops.length-1); const i=Math.floor(t), f=t-i, a=stops[i], b=stops[Math.min(i+1,stops.length-1)]; return a.map((v,k)=>Math.round(lerp(v,b[k],f))); };
  const rgb = (v) => `rgb(${v[0]},${v[1]},${v[2]})`;
  function restoreReloadState() {
    let navigationType=null, storage=null;
    try { navigationType=window.performance?.getEntriesByType?.('navigation')?.[0]?.type||null;storage=window.sessionStorage||null; } catch {}
    if(!storage)return {sessionId:null,pendingWorldCreate:null};
    if(!['reload','back_forward'].includes(navigationType)){
      try { storage.removeItem(ACTIVE_SESSION_STORAGE_KEY);storage.removeItem(PENDING_WORLD_STORAGE_KEY); } catch {}
      return {sessionId:null,pendingWorldCreate:null};
    }
    let sessionId=null,pendingWorldCreate=null;
    try {
      const storedSessionId=storage.getItem(ACTIVE_SESSION_STORAGE_KEY);
      if(typeof storedSessionId==='string'&&storedSessionId.length>0&&storedSessionId.length<=80)sessionId=storedSessionId;
      else if(storedSessionId)storage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
    } catch {}
    try {
      const pending=JSON.parse(storage.getItem(PENDING_WORLD_STORAGE_KEY)||'null');
      const replaceSessionIdValid=pending?.replace_session_id===undefined||pending?.replace_session_id===null||(typeof pending?.replace_session_id==='string'&&pending.replace_session_id.length>0&&pending.replace_session_id.length<=80);
      if(pending&&Number.isInteger(pending.seed)&&Number.isInteger(pending.width)&&Number.isInteger(pending.height)&&typeof pending.request_id==='string'&&pending.request_id.length>0&&pending.request_id.length<=80&&replaceSessionIdValid)pendingWorldCreate=pending;
      else storage.removeItem(PENDING_WORLD_STORAGE_KEY);
    } catch { try { storage.removeItem(PENDING_WORLD_STORAGE_KEY); } catch {} }
    return {sessionId,pendingWorldCreate};
  }
  function rememberSessionId(sessionId) {
    if(typeof sessionId!=='string'||!sessionId)return;
    try { window.sessionStorage?.setItem(ACTIVE_SESSION_STORAGE_KEY,sessionId); } catch {}
  }
  function persistPendingWorld(payload) {
    try { window.sessionStorage?.setItem(PENDING_WORLD_STORAGE_KEY,JSON.stringify(payload)); } catch {}
  }
  function clearPersistedPendingWorld() {
    try { window.sessionStorage?.removeItem(PENDING_WORLD_STORAGE_KEY); } catch {}
  }

  async function post(path, payload, timeoutMs=30000) {
    const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),timeoutMs);
    let responseComplete=false;
    const requestControlEpoch=state.controlEpoch;
    const requestControlGeneration=state.controlGeneration;
    const requestWorld=state.world;
    try {
      const headers={'Content-Type':'application/json','X-Page-ID':PAGE_ID};
      if(state.controlToken)headers['X-Control-Token']=state.controlToken;
      const res=await fetch(path, { method:'POST', headers, body:JSON.stringify(payload), signal:controller.signal });
      if (!res.ok) {
        const body=await res.json().catch(()=>({})),error=new Error(body.error||`请求失败 (${res.status})`);
        error.httpResponse=true;error.httpStatus=res.status;error.code=body.code;error.controlEpoch=body.control_epoch;
        error.controlOwner=body.control_page_id;error.controlReason=body.control_reason;error.controlConflict=body.code==='control_conflict';
        if(error.controlConflict) {
          if(requestWorld!==state.world||requestControlGeneration!==state.controlGeneration)error.staleControlResponse=true;
          else markControlConflict(error,requestControlEpoch);
        }
        throw error;
      }
      let body;
      try { body=await res.json(); }
      catch(e) {
        if(e?.name==='SyntaxError') {
          const error=new Error('仿真服务返回了无效的 JSON 响应');
          error.httpResponse=true;error.httpStatus=res.status;error.code='invalid_json_response';
          throw error;
        }
        throw e;
      }
      const sameAcceptedLease=typeof body.control_token==='string'&&body.control_token===state.controlToken&&
        body.control_epoch===state.controlEpoch&&state.controlOwner===PAGE_ID&&!state.controlBusy&&!state.controlLost;
      if(path!=='/api/world'&&requestControlGeneration!==state.controlGeneration&&!sameAcceptedLease) {
        const error=new Error('地图控制状态已变化，已忽略过期响应');error.httpResponse=true;error.code='control_conflict';error.controlConflict=true;error.staleControlResponse=true;throw error;
      }
      if(requestControlGeneration===state.controlGeneration&&typeof body.control_token==='string'&&Number.isInteger(body.control_epoch)&&body.control_epoch>=state.controlEpoch){
        if(state.controlToken!==body.control_token)state.controlGeneration++;
        state.controlToken=body.control_token;state.controlEpoch=body.control_epoch;state.controlOwner=PAGE_ID;state.controlBusy=false;state.controlLost=false;
        updateControlStates();
      }
      responseComplete=true;
      return body;
    } catch(e) {
      if(!e.httpResponse&&(e.name==='AbortError'||!responseComplete)){e.transportError=true;e.healthCheck=checkServer();}
      if(e.name==='AbortError'){const timeoutError=new Error(`请求超时（${Math.round(timeoutMs/1000)} 秒），请检查仿真服务后重试`);timeoutError.transportError=true;timeoutError.healthCheck=e.healthCheck;throw timeoutError;}
      throw e;
    } finally { clearTimeout(timeout); }
  }
  function markControlConflict(error,requestControlEpoch=state.controlEpoch) {
    const responseEpoch=Number.isInteger(error?.controlEpoch)?error.controlEpoch:requestControlEpoch;
    if(requestControlEpoch<state.controlEpoch||responseEpoch<state.controlEpoch)return;
    state.controlGeneration++;
    state.controlEpoch=Math.max(state.controlEpoch,responseEpoch);
    state.controlToken=null;
    state.controlOwner=error?.controlOwner||null;
    state.controlBusy=true;
    state.controlLost=true;
    discardUnconfirmedMovementDistance();
    stopFlight();
    canvas.blur?.();
    if(error?.controlReason==='page_released') {
      state.pageSessionReleased=true;state.controlBusy=false;state.controlLost=false;state.controlOwner=null;
      $('flightState').textContent='页面会话已释放';
      $('flightNote').textContent='当前页面的地图引用已释放，不能再取得控制权；刷新页面或重新生成地图后可继续操作。';
      $('statusMessage').textContent='当前页面会话引用已释放';
    } else if(state.controlOwner&&state.controlOwner!==PAGE_ID) {
      $('flightState').textContent='控制权被占用';
      $('flightNote').textContent='此地图正在由另一个页面控制；当前页面已停止写入。待控制权释放或租约到期后，可再次点击“开始自适应扫描”接管。';
      $('statusMessage').textContent='地图控制权已转移；当前页面进入只读状态';
    } else {
      state.controlBusy=false;state.controlLost=false;
      $('flightState').textContent='控制权已过期';
      $('flightNote').textContent='地图当前没有有效控制租约；点击“开始自适应扫描”即可重新接管。';
      $('statusMessage').textContent='地图控制租约已释放或过期';
    }
    updateControlStates();
    const world=state.world,revision=state.worldRevision;
    if(world&&!state.pageSessionReleased)void reconcileSessionSnapshot(world,revision);
  }
  async function postWorld(payload,revision) {
    const requestPayload={...payload,page_id:PAGE_ID};
    const capacityRetryDelays=[150,400,900];
    let capacityRetryCount=0;
    const requestWorld=async()=>{
      for(;;) {
        try { return await post('/api/world',requestPayload,60000); }
        catch(error) {
          const delay=capacityRetryDelays[capacityRetryCount];
          if(!isWorldCapacityRejection(error)||delay===undefined)throw error;
          capacityRetryCount++;
          await new Promise(resolve=>setTimeout(resolve,delay));
          if(revision!==state.worldRevision)throw error;
        }
      }
    };
    try { return await requestWorld(); }
    catch(error) {
      if(!error.transportError||revision!==state.worldRevision)throw error;
      await error.healthCheck;
      if(revision!==state.worldRevision)throw error;
      if(state.apiProtocolMismatch) {
        const mismatch=new Error('仿真服务接口版本不匹配，请重启仿真服务后刷新页面');
        mismatch.protocolMismatch=true; mismatch.preservePendingWorldCreate=true;
        throw mismatch;
      }
      if(state.connected!==true)throw error;
      return requestWorld();
    }
  }
  function isWorldCapacityRejection(error) {
    return error?.httpStatus===503&&error.message===WORLD_CAPACITY_ERROR;
  }
  function pointInWorld(point,world) {
    return Number.isInteger(point?.x)&&Number.isInteger(point?.y)&&point.x>=0&&point.y>=0&&point.x<world.width&&point.y<world.height;
  }
  function invalidWorldDataError(world) {
    const error=new Error('仿真服务返回的地图数据不完整或无效，请重启仿真服务后刷新页面');
    error.protocolMismatch=true;error.preservePendingWorldCreate=true;
    error.sessionId=world?.session_id;error.releaseToken=world?.release_token;
    return error;
  }
  async function releaseRejectedWorldSession(error) {
    if(typeof error.sessionId!=='string'||typeof error.releaseToken!=='string'||!error.releaseToken)return;
    try {
      await post('/api/session/release',{session_id:error.sessionId,release_token:error.releaseToken},3000);
    } catch {}
  }
  function validWorldResponse(world) {
    const weights=world?.layer_weights;
    // Older servers may omit this display-only weight; missing means zero to prevent line guidance.
    return Boolean(world&&typeof world==='object'&&typeof world.session_id==='string'&&world.session_id.length>0&&world.session_id.length<=80&&
      typeof world.release_token==='string'&&world.release_token.length>0&&Number.isInteger(world.width)&&world.width>=24&&world.width<=200&&
      Number.isInteger(world.height)&&world.height>=18&&world.height<=144&&world.cell_size_m===CELL_SIZE_METERS&&
      weights&&typeof weights==='object'&&!Array.isArray(weights)&&worldWeightNames.every(name=>
        (name==='dist_existing_line'&&!Object.prototype.hasOwnProperty.call(weights,name))||
        (Object.prototype.hasOwnProperty.call(weights,name)&&Number.isFinite(weights[name])&&weights[name]>=0&&weights[name]<=MAX_LAYER_WEIGHT))&&
      Object.values(weights).every(weight=>Number.isFinite(weight)&&weight>=0&&weight<=MAX_LAYER_WEIGHT)&&Array.isArray(world.layer_names)&&world.layer_names.length>0&&
      world.layer_names.length<=64&&world.layer_names.every(name=>typeof name==='string'&&name.length>0)&&
      new Set(world.layer_names).size===world.layer_names.length&&requiredWorldLayers.every(name=>world.layer_names.includes(name))&&
      pointInWorld(world.start,world)&&pointInWorld(world.goal,world));
  }
  function validSenseCell(cell,world) {
    const layers=cell?.layers;
    const maxSamples=MAX_SCAN_RECEIPTS*Math.ceil(Math.hypot(world.width-1,world.height-1)/SCAN_CENTER_SPACING);
    const samples=cell?.samples;
    if(!Number.isSafeInteger(samples)||samples<=0||samples>maxSamples)return false;
    const expectedConfidence=samples/(samples+2);
    return Boolean(pointInWorld(cell,world)&&Number.isFinite(cell.confidence)&&
      (cell.confidence===expectedConfidence||cell.confidence===Math.round(expectedConfidence*1000)/1000)&&layers&&typeof layers==='object'&&!Array.isArray(layers)&&
      Object.keys(layers).length===world.layer_names.length&&world.layer_names.every(name=>Object.prototype.hasOwnProperty.call(layers,name)&&
        Number.isFinite(layers[name])&&layers[name]>=0&&layers[name]<=1));
  }
  function validSenseResult(result,world) {
    return Boolean(result&&Array.isArray(result.cells)&&result.cells.every(cell=>validSenseCell(cell,world))&&
      (result.drone_position===undefined||pointInWorld(result.drone_position,world))&&
      (result.flight_distance===undefined||(Number.isFinite(result.flight_distance)&&result.flight_distance>=0))&&
      (result.scan_sequence===undefined||(Number.isSafeInteger(result.scan_sequence)&&result.scan_sequence>=0)));
  }
  function validSessionSnapshot(snapshot,world) {
    return Boolean(snapshot&&Array.isArray(snapshot.cells)&&Array.isArray(snapshot.route)&&Number.isSafeInteger(snapshot.scan_sequence)&&snapshot.scan_sequence>=0&&
      pointInWorld(snapshot.mission_start,world)&&pointInWorld(snapshot.mission_goal,world)&&pointInWorld(snapshot.drone_position,world)&&
      (snapshot.previous_drone_position===null||pointInWorld(snapshot.previous_drone_position,world))&&
      (snapshot.flight_distance===undefined||(Number.isFinite(snapshot.flight_distance)&&snapshot.flight_distance>=0))&&
      (snapshot.snapshot_revision===undefined||(Number.isSafeInteger(snapshot.snapshot_revision)&&snapshot.snapshot_revision>=0))&&
      (snapshot.scan_request_capacity_reached===undefined||typeof snapshot.scan_request_capacity_reached==='boolean')&&
      snapshot.cells.every(cell=>validSenseCell(cell,world))&&
      snapshot.route.every(point=>pointInWorld(point,world)));
  }
  function setConnectionStatus(connected, protocolMismatch=false) {
    state.connected=connected;
    state.apiProtocolMismatch=protocolMismatch;
    $('connection').textContent=connected?'仿真服务已连接':protocolMismatch?'仿真服务版本不兼容':'无法连接仿真服务';
    if(!state.world&&!state.worldLoading&&!state.worldLoadFailed){
      $('mapEmpty').classList.add('idle');
      $('mapStateTitle').textContent=connected?'仿真服务已连接':protocolMismatch?'仿真服务版本不兼容':'等待仿真服务连接';
      $('mapStateDetail').textContent=connected?'点击“随机地图”创建仿真地形。':protocolMismatch?'请重启当前仿真服务，再刷新此页面。':'服务恢复后可创建随机地图。';
    }
    document.querySelector('.top-status').classList.toggle('offline',!connected);
    document.querySelector('.top-status').classList.remove('pending');
    const serviceBadge=document.querySelector('.live-badge');
    serviceBadge.classList.toggle('offline',!connected);
    serviceBadge.classList.toggle('planner-unavailable',connected&&state.plannerReady===false);
    serviceBadge.classList.remove('pending');
    serviceBadge.setAttribute('title',connected?(state.plannerReady===false?state.plannerError:'仿真服务在线'):'仿真服务离线');
    serviceBadge.setAttribute('aria-label',connected?(state.plannerReady===false?`仿真服务在线，${state.plannerError}`:'仿真服务在线'):'仿真服务离线');
    $('serviceBadgeText').textContent=connected?(state.plannerReady===false?'规划不可用':'在线'):protocolMismatch?'版本不兼容':'离线';
  }
  function setPlannerStatus(ready,error=null) {
    state.plannerReady=typeof ready==='boolean'?ready:null;
    state.plannerError=state.plannerReady===false?String(error||'路径规划依赖未就绪，请检查 Python 环境。'):null;
    const badge=document.querySelector('.live-badge');
    const unavailable=state.connected===true&&state.plannerReady===false;
    badge.classList.toggle('planner-unavailable',unavailable);
    badge.setAttribute('title',unavailable?state.plannerError:state.connected?'仿真服务在线':'');
    badge.setAttribute('aria-label',unavailable?`仿真服务在线，${state.plannerError}`:state.connected?'仿真服务在线':'仿真服务状态');
    if(state.connected===true)$('serviceBadgeText').textContent=unavailable?'规划不可用':'在线';
  }
  function applyControlHealth(epoch,owner,controlled) {
    if(!Number.isInteger(epoch)||epoch<state.controlEpoch)return false;
    const oldOwner=state.controlOwner,oldBusy=state.controlBusy,oldLost=state.controlLost,oldEpoch=state.controlEpoch;
    if(state.pageSessionReleased) {
      state.controlEpoch=epoch;state.controlOwner=null;state.controlToken=null;state.controlBusy=false;state.controlLost=false;
      return false;
    }
    const wasCurrentController=state.controlOwner===PAGE_ID||Boolean(state.controlToken)||state.flying||Boolean(state.movement);
    state.controlEpoch=epoch;
    state.controlOwner=owner||null;
    if(controlled) {
      state.controlBusy=false;state.controlLost=false;
    } else if(owner&&owner!==PAGE_ID) {
      state.controlToken=null;
      state.controlBusy=true;state.controlLost=true;
      if(!oldBusy||!oldLost||oldOwner!==owner||epoch>oldEpoch)state.controlGeneration++;
      $('flightState').textContent='其他页面控制中';
      $('flightNote').textContent='另一个页面正在修改起终点、规划路线或执行扫描；当前页面保持只读。';
      $('statusMessage').textContent='地图由另一个页面控制；控制权释放后可重新接管';
      if(state.flying||state.movement) {
        discardUnconfirmedMovementDistance();
        stopFlight();
      }
    } else if(state.controlToken||state.flying||oldOwner===PAGE_ID) {
      state.controlGeneration++;
      state.controlToken=null;state.controlBusy=false;state.controlLost=false;
      discardUnconfirmedMovementDistance();
      stopFlight();
      $('flightState').textContent=owner===PAGE_ID?'控制租约已更新':'控制权已过期';
      $('flightNote').textContent=owner===PAGE_ID?'当前页面的控制租约已变化；点击“开始自适应扫描”重新确认。':'地图当前没有控制者；点击“开始自适应扫描”即可重新接管。';
      $('statusMessage').textContent=owner===PAGE_ID?'控制租约已变化，可重新确认':'地图控制租约已释放或过期';
    } else if(owner) {
      state.controlBusy=true;state.controlLost=true;
      if(!oldBusy||!oldLost||oldOwner!==owner||epoch>oldEpoch)state.controlGeneration++;
      $('flightState').textContent='其他页面控制中';
      $('flightNote').textContent='另一个页面正在修改起终点、规划路线或执行扫描；当前页面保持只读。';
      $('statusMessage').textContent='地图由另一个页面控制；控制权释放后可重新接管';
    } else {
      state.controlBusy=false;
      state.controlLost=false;
      if(oldBusy||oldLost||oldOwner)state.controlGeneration++;
      if((oldBusy||oldLost||oldOwner)&&!state.pageSessionReleased) {
        $('flightState').textContent='待命';
        $('flightNote').textContent='地图控制租约已释放或过期；点击“开始自适应扫描”即可重新接管。';
        $('statusMessage').textContent='地图控制权当前空闲';
      }
    }
    const releasedForeignControl=!controlled&&!owner&&oldOwner&&oldOwner!==PAGE_ID;
    if(releasedForeignControl)state.sessionSnapshotRequired=true;
    if(state.world&&!state.pageSessionReleased&&(state.sessionSnapshotRequired||(!controlled&&owner!==PAGE_ID&&(wasCurrentController||releasedForeignControl))))void reconcileSessionSnapshot(state.world,state.worldRevision);
    return oldOwner!==state.controlOwner||oldBusy!==state.controlBusy||oldLost!==state.controlLost;
  }
  function plannerUnavailableMessage() {
    return state.plannerError||'路径规划器依赖未就绪，请检查 README 中的 Python 依赖安装方式。';
  }
  function scanAvailabilityMessage() {
    if(state.sessionSnapshotRequired)return '共享地图状态仍在同步；同步完成前不能扫描或规划。';
    if(state.worldSessionUnconfirmed)return '地图会话仍在核验；核验完成前不能扫描或规划。';
    if(state.controlBusy) {
      if(state.scanRequestCapacityReached)return '另一页面正在控制地图；本地图扫描请求额度已用完，释放控制权后再重置地图或生成随机地图。';
      if(state.coverage>=1)return state.plannerReady===false
        ?'另一页面正在控制地图；扫描覆盖已完成，释放控制权后可生成随机地图，规划器恢复后可重新规划。'
        :'另一页面正在控制地图；扫描覆盖已完成，释放控制权后可重新规划或生成随机地图。';
      return '另一页面正在控制地图；可点击“开始自适应扫描”尝试接管，规划和地图替换需取得控制权。';
    }
    if(state.scanRequestCapacityReached)return '当前地图扫描请求额度已用完，请重置地图或生成随机地图后继续扫描。';
    if(state.coverage>=1)return '扫描覆盖已完成，无需继续扫描。';
    if(state.paused)return '扫描仍处于暂停状态；继续后可恢复当前任务。';
    return '地图扫描仍可继续。';
  }
  function showPlannerUnavailable() {
    $('planningBadge').textContent='规划器不可用';
    const scanStatus=scanAvailabilityMessage();
    $('statusMessage').textContent=`路径规划不可用；${scanStatus} ${plannerUnavailableMessage()}`;
    $('flightNote').textContent=`规划器不可用；${scanStatus} ${plannerUnavailableMessage()}`;
  }
  function showProtocolMismatch(message='仿真服务接口版本不匹配，请重启仿真服务后刷新页面') {
    const restoreFocus=Boolean(state.worldLoadingFocus&&(document.activeElement===document.body||document.activeElement===state.worldLoadingFocus));
    state.worldLoadFailed=true;
    state.worldSessionUnconfirmed=true;
    setConnectionStatus(false,true);
    preserveMovementScanTail();
    stopFlight();
    setWorldLoading(false);
    $('flightState').textContent='版本不兼容';
    $('flightNote').textContent=message;
    $('planningBadge').textContent='服务版本不兼容';
    $('statusMessage').textContent=message;
    if(state.world){
      $('mapEmpty').classList.add('hidden');
    } else {
      $('mapEmpty').classList.remove('hidden','failed');
      $('mapEmpty').classList.add('failed');
      $('mapStateTitle').textContent='仿真服务版本不兼容';
      $('mapStateDetail').textContent='请重启当前仿真服务，再刷新此页面。';
    }
    updateControlStates();
    if(restoreFocus)$('statusMessage').focus();
  }
  async function verifyWorldSession(world,revision) {
    const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),3000);
    const options={cache:'no-store',headers:{'X-Session-ID':world.session_id,'X-Page-ID':PAGE_ID},signal:controller.signal};
    if(state.controlToken)options.headers['X-Control-Token']=state.controlToken;
    try {
      const response=await fetch('/api/health',options);
      if(!response.ok){
        if(revision===state.worldRevision&&world===state.world)setConnectionStatus(false);
        return {status:'unknown',connected:false};
      }
      const health=await response.json();
      if(revision!==state.worldRevision||world!==state.world)return {status:'stale'};
      const protocolMismatch=health.ok!==false&&health.api_protocol!==API_PROTOCOL;
      const connected=health.ok!==false&&!protocolMismatch, instanceId=health.instance_id||null;
      const instanceChanged=connected&&Boolean(state.serverInstance&&instanceId&&state.serverInstance!==instanceId);
      const plannerReady=typeof health.planner_ready==='boolean'?health.planner_ready:instanceChanged?null:state.plannerReady;
      const plannerError=typeof health.planner_ready==='boolean'?(health.planner_error||null):instanceChanged?null:state.plannerError;
      if(connected&&instanceId)state.serverInstance=instanceId;
      setConnectionStatus(connected,protocolMismatch);
      if(connected)setPlannerStatus(plannerReady,plannerError);
      if(protocolMismatch)return {status:'incompatible',connected};
      if(!connected)return {status:'unknown',connected};
      if(instanceChanged||health.session_valid===false)return {status:'invalid',connected};
      if(health.session_valid===true)return {status:'valid',connected};
      return {status:'unknown',connected};
    } catch {
      if(revision===state.worldRevision&&world===state.world)setConnectionStatus(false);
      return {status:'unknown',connected:false};
    }
    finally { clearTimeout(timeout); }
  }
  function checkServer() {
    if(state.healthCheckPromise)return state.healthCheckPromise;
    const check=(async()=>{
      while(true) {
        const revision=state.worldRevision, world=state.world;
        const controlGeneration=state.controlGeneration;
        const plannerFailureRevision=state.plannerFailureRevision;
        const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),3000);
        let connected=false,protocolMismatch=false,instanceId=null,sessionValid=null,plannerReady=null,plannerError=null,controlEpoch=null,controlOwner=null,controlledByYou=false,snapshotRevision=null;
        const healthOptions={cache:'no-store',signal:controller.signal};
        if(world?.session_id){healthOptions.headers={'X-Session-ID':world.session_id,'X-Page-ID':PAGE_ID};if(state.controlToken)healthOptions.headers['X-Control-Token']=state.controlToken;}
        let plannerFieldProvided=false;
        try { const r=await fetch('/api/health',healthOptions); if(r.ok){const health=await r.json();protocolMismatch=health.ok!==false&&health.api_protocol!==API_PROTOCOL;connected=health.ok!==false&&!protocolMismatch;instanceId=health.instance_id||null;sessionValid=health.session_valid;plannerFieldProvided=typeof health.planner_ready==='boolean';plannerReady=plannerFieldProvided?health.planner_ready:null;plannerError=health.planner_error||null;controlEpoch=health.control_epoch;controlOwner=health.scan_control_page_id;controlledByYou=health.scan_controlled_by_you===true;snapshotRevision=health.snapshot_revision;} } catch {} finally { clearTimeout(timeout); }
        if(revision!==state.worldRevision||world!==state.world)continue;
        const controlChanged=controlGeneration===state.controlGeneration&&applyControlHealth(controlEpoch,controlOwner,controlledByYou);
        const sharedSnapshotChanged=Boolean(connected&&world&&!state.pageSessionReleased&&sessionValid!==false&&!controlledByYou&&(
          Number.isInteger(snapshotRevision)
            ?state.serverSnapshotVersion===null||snapshotRevision>state.serverSnapshotVersion
            :controlOwner&&controlOwner!==PAGE_ID
        ));
        if(sharedSnapshotChanged) {
          if(Number.isInteger(snapshotRevision))state.pendingServerSnapshotRevision=Math.max(state.pendingServerSnapshotRevision??-1,snapshotRevision);
          void reconcileSessionSnapshot(world,revision);
        }
        const instanceChanged=connected&&Boolean(state.serverInstance&&instanceId&&state.serverInstance!==instanceId);
        if(plannerFailureRevision!==state.plannerFailureRevision){plannerReady=state.plannerReady;plannerError=state.plannerError;}
        else if(connected&&!plannerFieldProvided){plannerReady=instanceChanged?null:state.plannerReady;plannerError=instanceChanged?null:state.plannerError;}
        const plannerChanged=connected&&(plannerReady!==state.plannerReady||plannerError!==state.plannerError);
        const previousPlannerReady=state.plannerReady;
        if(connected&&instanceId)state.serverInstance=instanceId;
        const sessionExpired=connected&&world&&sessionValid===false;
        const sessionReverified=Boolean(state.worldSessionUnconfirmed&&!state.pendingWorldCreate&&connected&&world&&sessionValid===true&&!instanceChanged);
        if(connected===state.connected&&protocolMismatch===state.apiProtocolMismatch&&!instanceChanged&&!sessionExpired&&!sessionReverified&&!plannerChanged&&!controlChanged&&!sharedSnapshotChanged)return;
        const previous=state.connected,previousProtocolMismatch=state.apiProtocolMismatch;
        if(protocolMismatch){showProtocolMismatch();return;}
        setConnectionStatus(connected,protocolMismatch);
        if(connected)setPlannerStatus(plannerReady,plannerError);
        if(!connected){
          $('statusMessage').textContent='仿真服务断开，正在重试连接';
          if(state.flying&&state.paused){
            $('flightState').textContent='已暂停';
            $('flightNote').textContent='仿真服务离线；扫描仍处于暂停状态，服务恢复后可继续。';
          }
          if(!world&&!state.worldLoadFailed){
            $('mapStateTitle').textContent='等待仿真服务连接';
            $('mapStateDetail').textContent='服务恢复后可创建随机地图。';
          }
        }
        else if(previousProtocolMismatch&&!world){$('mapStateTitle').textContent='服务已恢复';$('mapStateDetail').textContent='点击“随机地图”创建新地图。';$('statusMessage').textContent='仿真服务已更新兼容，请点击“随机地图”重新生成';}
        else if(previous===false&&!world&&$('mapStateTitle').textContent==='等待仿真服务连接'){$('mapStateTitle').textContent='服务已恢复';$('mapStateDetail').textContent='点击“随机地图”创建新地图。';$('statusMessage').textContent='仿真服务已恢复，请点击“随机地图”创建地图';}
        else if((instanceChanged&&state.world)||sessionExpired){state.worldSessionUnconfirmed=false;if(!state.sessionExpired)markSessionExpired();}
        else if(sessionReverified){
          state.worldSessionUnconfirmed=false;
          if(!state.worldLoading){
            if(state.scanRequestCapacityReached) {
              $('flightState').textContent='需要新地图';
              $('flightNote').textContent='旧地图会话已核验；当前地图扫描请求额度已用完，请重置地图或生成随机地图后继续扫描。';
              $('statusMessage').textContent='地图会话已核验，但扫描请求额度已用完';
            } else if(state.coverage>=1) {
              $('flightState').textContent='扫描完成';
              $('flightNote').textContent='旧地图会话已核验；扫描覆盖已完成，无需继续扫描。';
              $('statusMessage').textContent='地图会话已核验；扫描覆盖已完成';
            } else {
              $('flightState').textContent=state.flying?(state.paused?'已暂停':state.movement?'飞行中':'扫描中'):'待命';
              $('flightNote').textContent=state.flying&&state.paused?'旧地图会话已核验；扫描仍处于暂停状态，继续后可恢复当前任务。':state.flying?'旧地图会话已核验；当前扫描任务可继续。':'旧地图会话已核验，可以继续当前任务。';
              $('statusMessage').textContent=state.flying&&state.paused?'地图会话已核验；扫描仍处于暂停状态':'地图会话已核验，当前地图可继续';
            }
          }
        }
        else if(previous===false&&state.world&&state.flying&&state.paused&&!state.controlBusy&&!state.sessionSnapshotRequired&&!state.reconcilingScan){$('flightState').textContent='已暂停';$('flightNote').textContent='仿真服务已恢复；扫描保持暂停。';$('statusMessage').textContent='仿真服务已恢复；扫描保持暂停';}
        else if(previous===false)$('statusMessage').textContent=state.worldLoadFailed&&state.world&&!state.worldLoading?'仿真服务已恢复；当前地图已保留，可点击“随机地图”重试':!state.world&&$('mapStateTitle').textContent==='地图生成失败'?'仿真服务已恢复；请点击“随机地图”重新生成':'仿真服务已恢复';
        if(plannerChanged&&state.world){
          if(plannerReady===false){
            if(!instanceChanged&&!sessionExpired){
              if(state.pageSessionReleased)$('planningBadge').textContent='规划器不可用';
              else showPlannerUnavailable();
            }
          } else if(plannerReady===true&&previousPlannerReady===false){
            if($('planningBadge').textContent==='规划器不可用')$('planningBadge').textContent='尚未规划';
            if(!instanceChanged&&!sessionExpired&&!state.pageSessionReleased){
              if(state.sessionSnapshotRequired)$('flightState').textContent='地图状态待同步';
              else if(state.controlBusy)$('flightState').textContent='其他页面控制中';
              else if(state.scanRequestCapacityReached)$('flightState').textContent='需要新地图';
              else if(state.coverage>=1)$('flightState').textContent='扫描完成';
              else if(state.paused)$('flightState').textContent='已暂停';
              const scanStatus=scanAvailabilityMessage();
              $('flightNote').textContent=`规划器已恢复；${scanStatus}`;
              $('statusMessage').textContent=`路径规划器已恢复；${scanStatus}`;
            }
          }
        }
        updateControlStates();
        if(connected&&previous===false&&state.world&&state.coverage>=.20&&!state.route.length){
          if(state.planning||state.worldLoading)state.retryPlanPending=true;
          else planRoute();
        }
        if(plannerChanged&&plannerReady===true&&previousPlannerReady===false&&state.world&&!state.pendingScan&&state.coverage>=.20&&endpointsKnown()&&!state.route.length&&!state.planning){
          if(state.planAfterScan)consumeDeferredPlan(state.world);
          else planRoute();
        }
        return;
      }
    })();
    let shared;
    shared=check.finally(()=>{if(state.healthCheckPromise===shared)state.healthCheckPromise=null;});
    state.healthCheckPromise=shared;
    return shared;
  }
  function updateControlStates() {
    const offline=state.connected===false, awaitingHandshake=state.connected===null, loading=state.worldLoading, recovering=state.reconcilingScan;
    $('randomBtn').disabled=loading||offline||awaitingHandshake||recovering||state.controlBusy||state.startingFlight;
    $('resetBtn').disabled=loading||!state.world||offline||recovering||state.controlBusy||state.startingFlight||Boolean(state.pendingWorldCreate);
    $('startBtn').disabled=loading||offline||!state.world||state.sessionExpired||state.pageSessionReleased||state.scanRequestCapacityReached||state.sessionSnapshotRequired||state.worldSessionUnconfirmed||state.flying||state.startingFlight||(state.coverage>=1&&!state.pendingScan)||recovering;
    $('pauseBtn').disabled=loading||!state.flying||(offline&&state.paused)||recovering||state.startingFlight;
    const planLabel=state.flying?(state.paused?'已暂停自动优化':'扫描中自动优化'):state.planning?'路径计算中…':state.route.length?'重新规划路线':'规划初始路线';
    if($('planBtn').textContent!==planLabel)$('planBtn').textContent=planLabel;
    $('planBtn').disabled=loading||offline||state.controlBusy||state.startingFlight||state.plannerReady===false||state.sessionExpired||state.pageSessionReleased||state.sessionSnapshotRequired||state.worldSessionUnconfirmed||state.coverage<.20||!state.world||!endpointsKnown()||state.planning||state.flying||recovering||Boolean(state.pendingScan);
    const zoomUnavailable=loading||!state.world;
    $('zoomOutBtn').disabled=zoomUnavailable;
    $('zoomInBtn').disabled=zoomUnavailable;
    $('zoomFitBtn').disabled=zoomUnavailable;
    $('layerSelect').disabled=!state.world;
    $('zoomOutBtn').setAttribute('aria-disabled',String(zoomUnavailable||state.mapZoom<=1));
    $('zoomInBtn').setAttribute('aria-disabled',String(zoomUnavailable||state.mapZoom>=3));
    $('zoomFitBtn').setAttribute('aria-disabled',String(zoomUnavailable||state.mapZoom===1));
    const zoomText=`${state.mapZoom.toFixed(1)}×`, zoomLabel=`当前 ${state.mapZoom.toFixed(1)} 倍`;
    if($('mapZoomValue').textContent!==zoomText)$('mapZoomValue').textContent=zoomText;
    $('zoomOutBtn').setAttribute('aria-label',`缩小地图，${zoomLabel}`);
    $('zoomInBtn').setAttribute('aria-label',`放大地图，${zoomLabel}`);
    canvas.tabIndex=state.world&&!loading&&!recovering&&!state.controlBusy&&!state.startingFlight&&!state.sessionExpired&&!state.pageSessionReleased&&!state.sessionSnapshotRequired&&!state.worldSessionUnconfirmed?0:-1;
    const pointEditingDisabled=loading||!state.world||recovering||state.controlBusy||state.startingFlight||state.sessionExpired||state.pageSessionReleased||state.sessionSnapshotRequired||state.worldSessionUnconfirmed;
    const mapHint=!state.world?'生成地图后可选择起点或目标点。'
      :loading?'地图正在生成或恢复；完成前不能编辑端点。'
      :state.sessionSnapshotRequired||recovering?'共享地图状态正在同步；完成前不能编辑端点。'
      :state.worldSessionUnconfirmed?'地图会话结果待核验；核验完成前不能编辑端点。'
      :state.sessionExpired?'地图会话已过期；请生成随机地图后重新开始。'
      :state.pageSessionReleased?'当前页面的地图会话已释放；刷新页面或生成新地图后可继续编辑。'
      :state.controlBusy?'另一页面正控制地图；当前页只读，等待控制权释放后可编辑端点。'
      :state.startingFlight?'正在确认地图控制权；确认完成前不能编辑端点。'
      :'单击地图设置所选端点；画布获得焦点后用方向键移动坐标光标、Enter 设置；放大后视口获得焦点时方向键滚动地图；右键直接设置目标点';
    $('mapHint').textContent=mapHint;
    canvas.setAttribute('aria-disabled',String(pointEditingDisabled));
    if(pointEditingDisabled&&document.activeElement===canvas)$('statusMessage').focus();
    document.querySelectorAll('[data-point]').forEach(button=>button.disabled=pointEditingDisabled);
    for(const which of ['start','goal']) {
      $(`${which}X`).disabled=pointEditingDisabled; $(`${which}Y`).disabled=pointEditingDisabled; $(`apply${which==='start'?'Start':'Goal'}Coordinates`).disabled=pointEditingDisabled;
    }
  }
  function applyMapCanvasSize() {
    const viewport=$('mapViewport');
    const viewportRect=viewport.getBoundingClientRect();
    if(state.mapZoom===1) {
      canvas.style.width='100%'; canvas.style.height='100%';
    } else {
      canvas.style.width=`${Math.round(viewportRect.width*state.mapZoom)}px`;
      canvas.style.height=`${Math.round(viewportRect.height*state.mapZoom)}px`;
    }
    const size=mapViewportSize();
    state.mapViewportWidth=size.width;
    state.mapViewportHeight=size.height;
  }
  function mapViewportSize() {
    const viewport=$('mapViewport'), rect=viewport.getBoundingClientRect();
    return {width:viewport.clientWidth||rect.width,height:viewport.clientHeight||rect.height};
  }
  function mapViewAnchor() {
    if(!state.world)return null;
    const viewport=$('mapViewport'), geometry=JSON.parse(canvas.dataset.geometry||'{}'), size=mapViewportSize();
    if(!geometry.cell)return {x:state.world.width/2,y:state.world.height/2};
    return {
      x:(viewport.scrollLeft+(state.mapViewportWidth||size.width)/2-geometry.ox)/geometry.cell,
      y:(viewport.scrollTop+(state.mapViewportHeight||size.height)/2-geometry.oy)/geometry.cell,
    };
  }
  function restoreMapViewAnchor(anchor) {
    if(!anchor)return;
    const viewport=$('mapViewport'), geometry=JSON.parse(canvas.dataset.geometry||'{}'), size=mapViewportSize();
    if(!geometry.cell)return;
    viewport.scrollLeft=geometry.ox+anchor.x*geometry.cell-size.width/2;
    viewport.scrollTop=geometry.oy+anchor.y*geometry.cell-size.height/2;
  }
  function setMapZoom(value) {
    const zoom=Math.round(clamp(value,1,3)*2)/2, viewport=$('mapViewport');
    if(zoom===state.mapZoom)return;
    const anchor=mapViewAnchor();
    state.mapZoom=zoom;
    applyMapCanvasSize();
    updateControlStates(); draw();
    if(zoom===1){viewport.scrollLeft=0;viewport.scrollTop=0;}
    else restoreMapViewAnchor(anchor);
    refreshMapPointerReadout(true);
  }
  function resetMapZoom() { setMapZoom(1); }
  function resizeMap() {
    const anchor=state.mapZoom>1?mapViewAnchor():null;
    applyMapCanvasSize(); draw();
    restoreMapViewAnchor(anchor);
    refreshMapPointerReadout(true);
  }
  function revealKeyboardPoint(point) {
    if(state.mapZoom===1)return;
    const viewport=$('mapViewport'),geometry=JSON.parse(canvas.dataset.geometry||'{}');
    if(!geometry.cell)return;
    const margin=24, x=geometry.ox+(point.x+.5)*geometry.cell-(viewport.scrollLeft||0),
      y=geometry.oy+(point.y+.5)*geometry.cell-(viewport.scrollTop||0),
      width=viewport.clientWidth||viewport.getBoundingClientRect().width,
      height=viewport.clientHeight||viewport.getBoundingClientRect().height;
    if(x<margin)viewport.scrollLeft=Math.max(0,viewport.scrollLeft+x-margin);
    else if(x>width-margin)viewport.scrollLeft+=x-width+margin;
    if(y<margin)viewport.scrollTop=Math.max(0,viewport.scrollTop+y-margin);
    else if(y>height-margin)viewport.scrollTop+=y-height+margin;
  }
  function matrix(h,w,value=false) { return Array.from({length:h},()=>Array(w).fill(value)); }
  function isHardBlockedCell(layers,y,x) {
    const slope=clamp(layers.slope[y][x]||0), water=layers.water[y][x]||0, building=layers.building[y][x]||0;
    return Boolean(layers.hard_blocked?.[y]?.[x])||water>.90||slope>.96||building>.96;
  }
  function costGrid() {
    if(state.costGridCache)return state.costGridCache;
    const {layers,width,height}=state.world, out=matrix(height,width,0);
    for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
      const slope=clamp(layers.slope[y][x]||0), water=layers.water[y][x]||0, road=layers.road[y][x]||0,
        lu=layers.landuse[y][x]||0, building=layers.building[y][x]||0, vegetation=layers.vegetation[y][x]||0,
        protected=layers.protected[y][x]||0, seismic=layers.seismic[y][x]||0, landslide=layers.landslide[y][x]||0,
        typhoon=layers.typhoon[y][x]||0, ice=layers.ice[y][x]||0, lightning=layers.lightning[y][x]||0;
      // Preserve V5's 26-feature meaning: terrain, distance bands, land use, settlement and five hazards.
      const w=state.world.layer_weights||{}, weight=(name,fallback)=>Number(w[name]??fallback), normalized=(name)=>clamp(layers[name]?.[y]?.[x]||0);
      const hard=isHardBlockedCell(layers,y,x);
      const proximity=(name)=>1-normalized(name);
      const value=.08 + slope*weight('slope',.20) + (layers.tri?normalized('tri')*weight('tri',.06):0)
        + (layers.roughness_9?normalized('roughness_9')*weight('roughness_9',.05):0)
        + lu*weight('landuse',.12) + building*weight('building',.12) + vegetation*weight('vegetation',.06)
        + protected*weight('protected',.20) + seismic*weight('seismic',.04) + landslide*weight('landslide',.08)
        + typhoon*weight('typhoon',.03) + ice*weight('ice',.025) + lightning*weight('lightning',.02)
        + proximity('dist_water')*weight('dist_water',.12) + proximity('dist_fault')*weight('dist_fault',.08)
        + proximity('dist_railway')*weight('dist_railway',.025) + proximity('dist_existing_line')*weight('dist_existing_line',0)
        - proximity('dist_road')*weight('dist_road',.055);
      out[y][x]=hard?100:Math.max(.02,value);
    }
    state.costGridCache=out;
    return out;
  }
  function getCellColor(name,x,y,override) {
    const l=state.world.layers, v=name==='confidence'?state.confidence[y][x]:(override ?? l[name]?.[y]?.[x] ?? 0);
    if(name==='elevation') {
      return rgb(interp(terrainColors,(v-state.elevationRange[0])/Math.max(state.elevationRange[1]-state.elevationRange[0],.001)));
    }
    if(name==='cost') return v>=50?'#5a2734':rgb(interp(riskColors,clamp(v/1.05)));
    if(name==='slope') return rgb(interp(riskColors,clamp(v)));
    if(name==='water') return v>.35 ? '#2b86a3' : '#27392f';
    if(name==='road') return v>.48 ? '#e0bd78' : '#294437';
    if(name==='landuse') return landuseColors[Math.round(clamp(v)*5)];
    if(name==='building') return rgb(interp([[30,56,55],[87,133,106],[198,170,102],[217,87,69]],clamp(v)));
    if(name==='vegetation') return rgb(interp([[48,75,57],[68,125,76],[157,165,89],[210,185,109]],clamp(v)));
    if(name==='protected') return v>.45 ? 'rgba(186,90,133,.85)' : '#32473c';
    if(name==='hard_blocked') return v>.5 ? '#ab3f4b' : '#35473a';
    if(name==='confidence') return rgb(interp([[74,87,92],[72,155,144],[115,219,181]],clamp(v)));
    if(name.startsWith('dist_')) return rgb(interp([[58,111,119],[105,153,124],[210,184,103]],clamp(v)));
    if(['aspect_cos','aspect_sin','profile_curvature','plan_curvature','tpi_100','tpi_300','tpi_900'].includes(name)) return rgb(interp([[43,89,114],[86,138,128],[191,184,128],[148,95,102]],clamp(v)));
    if(['tri','roughness_3','roughness_9','roughness_27'].includes(name)) return rgb(interp([[47,93,72],[140,151,83],[200,140,78],[160,78,65]],clamp(v)));
    return rgb(interp(riskColors,clamp(v)));
  }
  function draw() {
    if(!state.world) return;
    const {width,height}=state.world, rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    if(!rect.width||!rect.height) return;
    const pixelRatio=Math.min(dpr,Math.sqrt(MAX_CANVAS_PIXELS/(rect.width*rect.height)));
    let pixelWidth=Math.round(rect.width*pixelRatio), pixelHeight=Math.round(rect.height*pixelRatio);
    if(pixelWidth*pixelHeight>MAX_CANVAS_PIXELS) {
      const correction=Math.sqrt(MAX_CANVAS_PIXELS/(pixelWidth*pixelHeight));
      pixelWidth=Math.floor(pixelWidth*correction);pixelHeight=Math.floor(pixelHeight*correction);
    }
    const scaleX=pixelWidth/rect.width,scaleY=pixelHeight/rect.height;
    if(canvas.width!==pixelWidth||canvas.height!==pixelHeight) { canvas.width=pixelWidth; canvas.height=pixelHeight; }
    const cell=Math.min((rect.width-48)/width,(rect.height-48)/height), ox=(rect.width-cell*width)/2, oy=(rect.height-cell*height)/2;
    const layer=$('layerSelect').value, cellPx=Math.max(1,cell+.3), costMap=layer==='cost'?costGrid():null;
    const signature=`${state.worldRevision}|${state.dataRevision}|${layer}|${rect.width}|${rect.height}|${dpr}|${pixelWidth}|${pixelHeight}`;
    if(signature!==state.baseSignature) {
      if(baseCanvas.width!==pixelWidth||baseCanvas.height!==pixelHeight) { baseCanvas.width=pixelWidth; baseCanvas.height=pixelHeight; }
      baseCtx.setTransform(scaleX,0,0,scaleY,0,0); baseCtx.globalAlpha=1; baseCtx.clearRect(0,0,rect.width,rect.height);
      for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
        baseCtx.fillStyle=getCellColor(layer,x,y,costMap?.[y]?.[x]); baseCtx.fillRect(ox+x*cell,oy+y*cell,cellPx,cellPx);
      }
      // A thin shaded relief cue on top of every layer makes ridges and valleys legible.
      const elev=state.world.layers.elevation;
      baseCtx.globalAlpha=.13; baseCtx.strokeStyle='#e6e2cb'; baseCtx.lineWidth=.5;
      for(let y=1;y<height-1;y+=4) for(let x=1;x<width-1;x+=4) {
        if(state.observed[y][x]&&state.observed[y-1][x]&&state.observed[y][x-1]&&elev[y][x]>elev[y-1][x]&&elev[y][x]>elev[y][x-1]) { baseCtx.beginPath(); baseCtx.moveTo(ox+x*cell,oy+y*cell); baseCtx.lineTo(ox+(x+2)*cell,oy+(y-1)*cell); baseCtx.stroke(); }
      }
      baseCtx.globalAlpha=1;
      // Unobserved terrain remains hidden irrespective of the selected layer.
      baseCtx.fillStyle='#081116';
      for(let y=0;y<height;y++) for(let x=0;x<width;x++) if(!state.observed[y][x]) baseCtx.fillRect(ox+x*cell,oy+y*cell,cellPx,cellPx);
      baseCtx.beginPath();
      for(let x=0;x<=width;x++){baseCtx.moveTo(ox+x*cell,oy);baseCtx.lineTo(ox+x*cell,oy+height*cell);}
      for(let y=0;y<=height;y++){baseCtx.moveTo(ox,oy+y*cell);baseCtx.lineTo(ox+width*cell,oy+y*cell);}
      baseCtx.strokeStyle='rgba(113,151,151,.07)';baseCtx.lineWidth=.65;baseCtx.stroke();
      if(state.coverage===0) {
        baseCtx.fillStyle='rgba(164,190,185,.48)';baseCtx.font='12px sans-serif';baseCtx.textAlign='center';baseCtx.fillText('未探测区域',ox+width*cell/2,oy+height*cell/2-5);
        baseCtx.fillStyle='rgba(133,159,158,.38)';baseCtx.font='10px sans-serif';baseCtx.fillText('启动扫描后逐步揭示地形',ox+width*cell/2,oy+height*cell/2+14);
      }
      state.baseSignature=signature;
    }
    ctx.setTransform(scaleX,0,0,scaleY,0,0); ctx.clearRect(0,0,rect.width,rect.height); ctx.drawImage(baseCanvas,0,0,rect.width,rect.height);
    // Keep the last scanned swath visible while the UAV moves between scan points.
    if(state.lastDrone) {
      const p=state.lastDrone, px=ox+(p.x+.5)*cell, py=oy+(p.y+.5)*cell;
      ctx.beginPath(); ctx.arc(px,py,Math.max(5,cell*4.6),0,Math.PI*2); ctx.strokeStyle='rgba(62,220,190,.62)'; ctx.lineWidth=1.5; ctx.stroke();
    }
    if(state.route.length>1) {
      ctx.beginPath(); state.route.forEach((p,i)=>{const px=ox+(p.x+.5)*cell,py=oy+(p.y+.5)*cell;i?ctx.lineTo(px,py):ctx.moveTo(px,py)});
      ctx.strokeStyle='rgba(246,197,101,.22)';ctx.lineWidth=6;ctx.lineJoin='round';ctx.stroke();ctx.strokeStyle='#f6c565';ctx.lineWidth=2.2;ctx.stroke();
    }
    if(state.scanTrail.length>1) {
      const trail=state.scanTrail;
      ctx.beginPath();
      trail.forEach((p,index)=>{const px=ox+(p.x+.5)*cell,py=oy+(p.y+.5)*cell;index?ctx.lineTo(px,py):ctx.moveTo(px,py);});
      ctx.strokeStyle='rgba(90,240,209,.42)';ctx.lineWidth=1.5;ctx.lineJoin='round';ctx.stroke();
    }
    if(state.movement&&state.dronePosition) {
      const from=state.movement.from,to=state.dronePosition;
      if(Math.hypot(to.x-from.x,to.y-from.y)>.05) {
        ctx.save();ctx.beginPath();ctx.moveTo(ox+(from.x+.5)*cell,oy+(from.y+.5)*cell);ctx.lineTo(ox+(to.x+.5)*cell,oy+(to.y+.5)*cell);
        ctx.strokeStyle='rgba(90,240,209,.9)';ctx.lineWidth=2.2;ctx.lineCap='round';ctx.stroke();
        ctx.restore();
      }
    }
    canvas.dataset.geometry=JSON.stringify({cell,ox,oy,width,height});
    drawScale(cell,width);
    const coincidentEndpoints=state.world.start.x===state.world.goal.x&&state.world.start.y===state.world.goal.y;
    const startCenter={x:ox+(state.world.start.x+.5)*cell,y:oy+(state.world.start.y+.5)*cell};
    const goalCenter={x:ox+(state.world.goal.x+.5)*cell,y:oy+(state.world.goal.y+.5)*cell};
    const endpointDistance=Math.hypot(goalCenter.x-startCenter.x,goalCenter.y-startCenter.y);
    const nearbyEndpoints=!coincidentEndpoints&&endpointDistance<Math.max(14,Math.max(6,cell*1.65)*2+2);
    const coincidentOffsets=coincidentEndpoints?coincidentLabelOffsets(state.world.start,cell,ox,oy,rect.width,rect.height):null;
    const nearbyLabelOffsets=nearbyEndpoints?separateEndpointLabelOffsets(state.world.start,state.world.goal,cell,ox,oy,rect.width,rect.height):null;
    if(coincidentEndpoints)drawCoincidentPoints(state.world.start);
    else if(nearbyEndpoints){const radius=endpointDistance*.25;drawPoint(state.world.start,'#69d5bd',radius);drawPoint(state.world.goal,'#f48770',radius);}
    else{drawPoint(state.world.start,'#69d5bd');drawPoint(state.world.goal,'#f48770');}
    if(coincidentOffsets){drawPointLeaderLine(state.world.start,coincidentOffsets[0],'#69d5bd');drawPointLeaderLine(state.world.goal,coincidentOffsets[1],'#f48770');}
    else if(nearbyEndpoints){drawPointLeaderLine(state.world.start,nearbyLabelOffsets[0],'#69d5bd');drawPointLeaderLine(state.world.goal,nearbyLabelOffsets[1],'#f48770');}
    if(state.dronePosition) {
      const p=state.dronePosition, px=ox+(p.x+.5)*cell, py=oy+(p.y+.5)*cell;
      const heading=state.movement?Math.atan2(state.movement.to.y-state.movement.from.y,state.movement.to.x-state.movement.from.x)+Math.PI/4:Math.PI/4;
      ctx.save(); ctx.translate(px,py); ctx.rotate(heading); ctx.fillStyle='#5af0d1'; ctx.shadowColor='#44edce'; ctx.shadowBlur=14; ctx.fillRect(-5,-5,10,10); ctx.restore();
    }
    if(coincidentOffsets){drawPointLabel(state.world.start,'A',coincidentOffsets[0],'#69d5bd',false);drawPointLabel(state.world.goal,'B',coincidentOffsets[1],'#f48770',false);}
    else if(nearbyEndpoints){drawPointLabel(state.world.start,'A',nearbyLabelOffsets[0],'#69d5bd',false);drawPointLabel(state.world.goal,'B',nearbyLabelOffsets[1],'#f48770',false);}
    else{drawPointLabel(state.world.start,'A');drawPointLabel(state.world.goal,'B');}
    if(state.keyboardPoint&&document.activeElement===canvas)drawKeyboardPoint(state.keyboardPoint,cell,ox,oy);
    updateLayerLegend();
  }
  function drawScale(cell,width) {
    const metersPerCell=state.world.cell_size_m||100, maxPixels=Math.min(cell*width*.24,100), maxMeters=maxPixels/cell*metersPerCell;
    const power=10**Math.floor(Math.log10(maxMeters||1)), amount=[1,2,5,10].filter(n=>n*power<=maxMeters).pop()||1;
    const meters=amount*power;
    $('scaleBar').style.width=`${meters/metersPerCell*cell}px`;
    $('scaleLabel').textContent=meters>=1000?`${Number((meters/1000).toFixed(2))} km`:`${Math.round(meters)} m`;
  }
  function drawKeyboardPoint(p,cell,ox,oy) {
    const x=ox+(p.x+.5)*cell,y=oy+(p.y+.5)*cell,r=Math.min(18,Math.max(6,cell*1.5));
    ctx.save();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.setLineDash([3,2]);ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(x-r-3,y);ctx.lineTo(x+r+3,y);ctx.moveTo(x,y-r-3);ctx.lineTo(x,y+r+3);ctx.stroke();ctx.restore();
  }
  function updateLayerLegend() {
    const name=$('layerSelect').value,label=layerLabels[name]||name;
    const thresholds={water:.35,road:.48,protected:.45,hard_blocked:.5};
    const thresholdClasses={water:['水域','非水域'],road:['道路','非道路'],protected:['保护区','普通区域'],hard_blocked:['硬约束禁行区','非硬约束禁行区']};
    if(!(name in state.layerRangeCache)){
      const values=name==='confidence'?state.confidence:(name==='cost'?costGrid():state.world.layers[name]);
      let min=Infinity,max=-Infinity;
      for(let y=0;y<state.world.height;y++)for(let x=0;x<state.world.width;x++)if(state.observed[y][x]){
        const value=values?.[y]?.[x];if(Number.isFinite(value)&&!(name==='cost'&&value>=50)){min=Math.min(min,value);max=Math.max(max,value);}
      }
      state.layerRangeCache[name]=min===Infinity?null:[min,max];
    }
    const range=state.layerRangeCache[name],note=name==='cost'?' · 可通行色带上限 1.05 · 深酒红：硬约束禁行（100标记）':name==='landuse'?' · 6类编码（0–1）':name==='confidence'?' · 重复采样置信度 n/(n+2)':name in thresholds?` · 归一化值 · 分色阈值 ${thresholds[name].toFixed(2)}`:['dist_road','dist_water'].includes(name)?' · 反向归一化强度 · 非距离值':['dist_existing_line','dist_railway','dist_fault'].includes(name)?' · 归一化合成代理 · 非米制距离':name==='elevation'?' · 按已观测极值动态拉伸 · 归一化模拟值 · 无物理高程单位':derivedProxyLayers.includes(name)?' · 合成归一化代理 · 非标准DEM派生量':' · 归一化模拟值 · 无量纲';
    const text=range?`已观测值域 · ${range[0].toFixed(2)} – ${range[1].toFixed(2)}${note}`:name==='cost'&&state.coverage>0?`已观测区域暂无可通行代价${note}`:`等待扫描 · 仅显示已观测数据${note}`;
    const domain=name==='cost'?[0,1.05]:name==='elevation'?(range||[0,1]):[0,1];
    const palette=name==='elevation'?terrainColors.map(rgb):name==='water'?['#27392f','#2b86a3']:name==='road'?['#294437','#e0bd78']
      :name==='landuse'?landuseColors
      :name==='building'?['#1e3837','#57856a','#c6aa66','#d95745']:name==='vegetation'?['#304b39','#447d4c','#9da559','#d2b96d']
      :name==='protected'?['#32473c','rgba(186,90,133,.85)']:name==='hard_blocked'?['#35473a','#ab3f4b']
      :name==='confidence'?['#4a575c','#489b90','#73dbb5']
      :['aspect_cos','aspect_sin','profile_curvature','plan_curvature','tpi_100','tpi_300','tpi_900'].includes(name)?['#2b5972','#568a80','#bfb880','#945f66']
      :['tri','roughness_3','roughness_9','roughness_27'].includes(name)?['#2f5d48','#8c9753','#c88c4e','#a04e41']
      :name.startsWith('dist_')?['#3a6f77','#69997c','#d2b867']:riskColors.map(rgb);
    const signature=`${name}|${text}|${domain.join(',')}|${palette.join(',')}`;
    if(signature===state.layerLegendSignature)return;
    state.layerLegendSignature=signature;
    $('layerLegendName').textContent=label;$('layerLegendText').textContent=text;
    let gradient;
    if(name in thresholds)gradient=`linear-gradient(90deg,${palette[0]} ${thresholds[name]*100}%,${palette[palette.length-1]} ${thresholds[name]*100}%)`;
    else if(name==='landuse'){
      const stops=[`${palette[0]} 0%`];
      for(let i=1;i<palette.length;i++){const point=(i-.5)/(palette.length-1)*100;stops.push(`${palette[i-1]} ${point}%`,`${palette[i]} ${point}%`);}
      stops.push(`${palette[palette.length-1]} 100%`);gradient=`linear-gradient(90deg,${stops.join(',')})`;
    } else if(name==='elevation'&&range&&range[0]===range[1])gradient=`linear-gradient(90deg,${palette[0]},${palette[0]})`;
    else gradient=`linear-gradient(90deg,${palette.join(',')})`;
    $('layerGradient').style.background=gradient;
    const colorDomain=name==='elevation'&&range?`颜色按已观测值域 ${range[0].toFixed(2)} 至 ${range[1].toFixed(2)} 拉伸`:`颜色映射值域 ${domain[0].toFixed(2)} 至 ${domain[1].toFixed(2)}`;
    const observedRange=range&&name!=='elevation'?`，已观测数据值域 ${range[0].toFixed(2)} 至 ${range[1].toFixed(2)}`:
      !range&&name==='cost'&&state.coverage>0?'，已观测格均为硬约束禁行区并以深酒红标示':!range?'，尚无已观测数据':'';
    const special=name==='cost'?'，代价值 ≥50 表示硬约束禁行区，以深酒红单独标示（不属于渐变范围）':'';
    const threshold=name in thresholds?`，大于 ${thresholds[name].toFixed(2)} 为${thresholdClasses[name][0]}，不大于该值为${thresholdClasses[name][1]}`:'';
    const landuse=name==='landuse'?'，0 至 1 按最近的 0 至 5 类编码着色（共 6 类）':'';
    $('layerGradient').setAttribute('aria-label',`${label}，${colorDomain}${observedRange}${special}${threshold}${landuse}`);
    $('layerRangeMin').textContent=domain[0].toFixed(2);$('layerRangeMax').textContent=domain[1].toFixed(2);
  }
  function endpointMarkerRadius(cell) { return Math.min(18,Math.max(6,cell*1.65)); }
  function drawPoint(p,color,radius=null) { if(!p)return; const g=JSON.parse(canvas.dataset.geometry||'{}');if(!g.cell)return;const x=g.ox+(p.x+.5)*g.cell,y=g.oy+(p.y+.5)*g.cell,r=Math.min(18,radius??endpointMarkerRadius(g.cell));ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.lineWidth=radius!==null&&radius<3?1:2;ctx.strokeStyle='#0d171c';ctx.stroke(); }
  function drawCoincidentPoints(p) {
    const g=JSON.parse(canvas.dataset.geometry||'{}');if(!g.cell)return;
    const x=g.ox+(p.x+.5)*g.cell,y=g.oy+(p.y+.5)*g.cell,r=endpointMarkerRadius(g.cell);
    for(const [from,to,color] of [[Math.PI/2,Math.PI*1.5,'#69d5bd'],[-Math.PI/2,Math.PI/2,'#f48770']]){
      ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(x,y);ctx.arc(x,y,r,from,to);ctx.closePath();ctx.fill();
    }
    ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.lineWidth=2;ctx.strokeStyle='#0d171c';ctx.stroke();
  }
  function coincidentLabelOffsets(p,cell,ox,oy,width,height) {
    const x=ox+(p.x+.5)*cell,y=oy+(p.y+.5)*cell,distance=endpointMarkerRadius(cell)+10,half=6;
    const directions=[[-1,0],[1,0],[0,-1],[0,1],[-.707,-.707],[.707,-.707],[-.707,.707],[.707,.707]];
    const available=directions.map(([dx,dy])=>({x:dx*distance,y:dy*distance})).filter(offset=>
      x+offset.x-half>=0&&x+offset.x+half<=width&&y+offset.y-half>=0&&y+offset.y+half<=height);
    return available.length>=2?available.slice(0,2):[{x:-distance,y:0},{x:distance,y:0}];
  }
  function drawPointLeaderLine(p,offset,color) {
    if(!p||!offset||(!offset.x&&!offset.y))return;const g=JSON.parse(canvas.dataset.geometry||'{}');if(!g.cell)return;
    ctx.beginPath();ctx.moveTo(g.ox+(p.x+.5)*g.cell,g.oy+(p.y+.5)*g.cell);ctx.lineTo(g.ox+(p.x+.5)*g.cell+offset.x,g.oy+(p.y+.5)*g.cell+offset.y);ctx.strokeStyle=color;ctx.lineWidth=1;ctx.stroke();
  }
  function separateEndpointLabelOffsets(start,goal,cell,ox,oy,width,height) {
    const center=point=>({x:ox+(point.x+.5)*cell,y:oy+(point.y+.5)*cell}),a=center(start),b=center(goal),directions=[
      [-1,0],[1,0],[0,-1],[0,1],[-.707,-.707],[.707,-.707],[-.707,.707],[.707,.707],
    ],candidates=[{x:0,y:0}];
    for(const distance of [10,14,20,28])for(const [dx,dy] of directions)candidates.push({x:dx*distance,y:dy*distance});
    let best=null,bestCost=Infinity;
    for(const first of candidates)for(const second of candidates) {
      const x1=a.x+first.x,y1=a.y+first.y,x2=b.x+second.x,y2=b.y+second.y;
      if(x1<6||x1>width-6||y1<6||y1>height-6||x2<6||x2>width-6||y2<6||y2>height-6||
         (Math.abs(x1-a.x)<14&&Math.abs(y1-a.y)<14)||(Math.abs(x1-b.x)<14&&Math.abs(y1-b.y)<14)||
         (Math.abs(x2-a.x)<14&&Math.abs(y2-a.y)<14)||(Math.abs(x2-b.x)<14&&Math.abs(y2-b.y)<14)||
         (Math.abs(x2-x1)<14&&Math.abs(y2-y1)<14))continue;
      const cost=Math.hypot(first.x,first.y)+Math.hypot(second.x,second.y);
      if(cost<bestCost){best=[first,second];bestCost=cost;}
    }
    return best||coincidentLabelOffsets(start,cell,ox,oy,width,height);
  }
  function drawPointLabel(p,label,offset,color,drawLeader=true) {
    if(!p)return;const g=JSON.parse(canvas.dataset.geometry||'{}');if(!g.cell)return;
    const x=g.ox+(p.x+.5)*g.cell+(offset?.x||0),y=g.oy+(p.y+.5)*g.cell+(offset?.y||0);
    if(offset){if(drawLeader)drawPointLeaderLine(p,offset,color);ctx.fillStyle=color;ctx.fillRect(x-6,y-6,12,12);ctx.strokeStyle='#0d171c';ctx.lineWidth=1;ctx.strokeRect(x-6,y-6,12,12);}
    ctx.font='bold 9px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#0c161a';ctx.fillText(label,x,y+.5);
  }
  function coordinates(p) { return `X ${p.x} · Y ${p.y}`; }
  function announceKeyboardPoint(p,message='键盘光标') {
    const layer=$('layerSelect').value,label=layerLabels[layer]||layer;
    const observed=Boolean(state.world&&state.observed[p.y]?.[p.x]);
    let cellDescription=`未探测（迷雾），${label}值不可用`;
    if(observed) {
      const value=layer==='confidence'?state.confidence[p.y]?.[p.x]:layer==='cost'?costGrid()[p.y]?.[p.x]:state.world.layers[layer]?.[p.y]?.[p.x];
      if(Number.isFinite(value)) {
        let classLabel='';
        if(layer==='landuse')classLabel=`土地利用第 ${Math.round(clamp(value)*5)+1} 类`;
        else if(layer==='water')classLabel=value>.35?'水域':'非水域';
        else if(layer==='road')classLabel=value>.48?'道路':'非道路';
        else if(layer==='protected')classLabel=value>.45?'保护区':'普通区域';
        else if(layer==='hard_blocked')classLabel=value>.5?'硬约束禁行区':'非硬约束禁行区';
        else if(layer==='cost'&&value>=50)classLabel='硬约束禁行';
        cellDescription=`已探测，${classLabel?`${classLabel}，${value.toFixed(2)}`:`${label} ${value.toFixed(2)}`}`;
      } else cellDescription=`已探测，${label}无数值`;
    }
    const pointRole=message==='当前格'?'':`${state.activePoint==='start'?'起点':'目标点'} `;
    const announcement=`${message}：${pointRole}${coordinates(p)}；${cellDescription}`;
    if($('keyboardCoordinates').textContent!==announcement)$('keyboardCoordinates').textContent=announcement;
  }
  function updatePoints(onlyWhich=null) {
    for(const which of onlyWhich?[onlyWhich]:['start','goal']) {
      const point=state.world?.[which];
      $(`${which}X`).value=point?String(point.x):''; $(`${which}Y`).value=point?String(point.y):'';
      if(state.world){$(`${which}X`).setAttribute('max',String(state.world.width-1));$(`${which}Y`).setAttribute('max',String(state.world.height-1));}
    }
  }
  function applyCoordinateInputs(which) {
    if(state.pageSessionReleased||state.sessionSnapshotRequired)return false;
    if(!state.world||state.worldLoading||state.reconcilingScan||state.controlBusy||state.startingFlight||state.sessionExpired||state.worldSessionUnconfirmed)return false;
    const xValue=$(`${which}X`).value,yValue=$(`${which}Y`).value,x=Number(xValue),y=Number(yValue),{width,height}=state.world;
    if(xValue===''||yValue===''||!Number.isInteger(x)||!Number.isInteger(y)||x<0||x>=width||y<0||y>=height) {
      $('statusMessage').textContent=`请输入有效的${which==='start'?'起点':'目标点'}格坐标（X：0–${width-1}，Y：0–${height-1}）`;
      return false;
    }
    if(state.world[which].x===x&&state.world[which].y===y){selectPoint(which);return true;}
    state.world[which]={x,y}; state.keyboardPoint={x,y}; selectPoint(which); announceKeyboardPoint(state.world[which],'已设置'); endpointsChanged(which);
    return true;
  }
  function applySenseResult(result,world) {
    if(world!==state.world||state.sessionExpired)return false;
    if(!validSenseResult(result,world))throw new Error('服务返回的扫描数据不完整或无效');
    const keyboardPoint=state.keyboardPoint, keyboardPointWasObserved=keyboardPoint&&state.observed[keyboardPoint.y]?.[keyboardPoint.x];
    const previouslyBlocked=knownBlockedEndpoints();
    let updated=false, keyboardPointUpdated=false;
    for(const cell of result.cells) {
      const samples=cell.samples;
      if(samples<state.samples[cell.y][cell.x])continue;
      if(keyboardPoint&&cell.x===keyboardPoint.x&&cell.y===keyboardPoint.y)keyboardPointUpdated=true;
      const unchanged=state.observed[cell.y][cell.x]&&samples===state.samples[cell.y][cell.x]&&
        cell.confidence===state.confidence[cell.y][cell.x]&&
        Object.entries(cell.layers).every(([name,value])=>state.world.layers[name][cell.y][cell.x]===value);
      if(unchanged)continue;
      updated=true;
      state.observed[cell.y][cell.x]=true;
      state.confidence[cell.y][cell.x]=cell.confidence;
      if(samples!==null)state.samples[cell.y][cell.x]=samples;
      for(const [name,value] of Object.entries(cell.layers)) state.world.layers[name][cell.y][cell.x]=value;
    }
    if(updated){state.dataRevision++;state.costGridCache=null;state.layerRangeCache={};}
    const newlyBlocked=knownBlockedEndpoints().filter(which=>!previouslyBlocked.includes(which));
    if(newlyBlocked.length)publishBlockedEndpointWarning();
    if(keyboardPointUpdated&&document.activeElement===canvas&&state.observed[keyboardPoint.y]?.[keyboardPoint.x])announceKeyboardPoint(keyboardPoint,keyboardPointWasObserved?'当前格':undefined);
    return true;
  }
  function staleSessionSnapshot(snapshot) {
    if(!Number.isInteger(snapshot?.snapshot_revision))return false;
    const newest=Math.max(state.serverSnapshotVersion??-1,state.pendingServerSnapshotRevision??-1);
    return snapshot.snapshot_revision<newest;
  }
  function applySessionSnapshot(snapshot,world) {
    if(world!==state.world||!validSessionSnapshot(snapshot,world)||staleSessionSnapshot(snapshot))return false;
    if(Number.isInteger(snapshot.snapshot_revision)) {
      state.serverSnapshotVersion=snapshot.snapshot_revision;
      if(Number.isInteger(state.pendingServerSnapshotRevision)&&snapshot.snapshot_revision>=state.pendingServerSnapshotRevision) {
        state.pendingServerSnapshotRevision=null;
        state.pendingServerSnapshotRetryRevision=null;
      }
    }
    state.planRevision++;state.planning=false;state.planQueued=false;state.retryPlanPending=false;
    if(typeof snapshot.scan_request_capacity_reached==='boolean')state.scanRequestCapacityReached=snapshot.scan_request_capacity_reached;
    applySenseResult(snapshot,world);
    world.start={...snapshot.mission_start};world.goal={...snapshot.mission_goal};
    state.endpointRevision++;state.missionSyncedRevision=state.endpointRevision;
    state.dronePosition={...snapshot.drone_position};
    if(Number.isFinite(snapshot.flight_distance)) {
      state.flightDistance=snapshot.flight_distance;
      $('flightDistanceText').textContent=`${(state.flightDistance*(world.cell_size_m||100)/1000).toFixed(2)} km`;
    }
    state.previousDronePosition=snapshot.previous_drone_position?{...snapshot.previous_drone_position}:null;
    state.lastDrone={...snapshot.drone_position};
    state.scanTrail=state.previousDronePosition?[{...state.previousDronePosition},{...snapshot.drone_position}]:[{...snapshot.drone_position}];
    state.route=snapshot.route.map(point=>({...point}));state.scanPath=scanPattern();
    if(state.route.length) {
      const routeDistance=state.route.slice(1).reduce((sum,point,index)=>sum+Math.hypot(
        point.x-state.route[index].x,point.y-state.route[index].y,
      ),0);
      $('distanceMetric').textContent=((routeDistance*(world.cell_size_m||100))/1000).toFixed(2);
      $('costMetric').textContent='—';
      $('searchMetric').textContent=state.route.length.toLocaleString();
      $('planningBadge').textContent=state.plannerReady===false?'规划器不可用':'已同步路线';
    } else {
      $('distanceMetric').textContent='—';$('costMetric').textContent='—';$('searchMetric').textContent='—';
      $('planningBadge').textContent=state.plannerReady===false?'规划器不可用':'尚未规划';
    }
    publishBlockedEndpointWarning();
    state.pendingScan=null;updatePoints();refreshCoverage();
    state.scanSincePlan=Math.floor(state.coverage/.20);draw();updateControlStates();
    return true;
  }
  function announceScanRecovery(world) {
    if(state.coverage>=1)return;
    const position=state.dronePosition||world.start;
    $('scanAnnouncement').textContent=`上一段扫描数据已恢复：无人机位置 X ${Math.round(position.x)}、Y ${Math.round(position.y)}；累计航程 ${(state.flightDistance*(world.cell_size_m||100)/1000).toFixed(2)} 公里；覆盖率 ${$('coverageText').textContent}`;
    $('flightNote').textContent=`上一段扫描数据已恢复；无人机位于 X ${Math.round(position.x)}、Y ${Math.round(position.y)}；${state.connected===false?'服务离线，连接恢复后可继续扫描。':'可以继续扫描。'}`;
  }
  function reconcileSessionSnapshot(world,revision) {
    if(!world||world!==state.world||revision!==state.worldRevision||state.pageSessionReleased)return Promise.resolve(false);
    if(state.sessionSnapshotPromise)return state.sessionSnapshotPromise;
    state.sessionSnapshotRequired=true;
    const controlGeneration=state.controlGeneration;
    const requestRevision=++state.sessionSnapshotRevision;
    if(!state.controlBusy&&!state.controlToken) {
      $('flightState').textContent='同步地图状态';
      $('flightNote').textContent='正在同步共享地图数据；同步完成后可继续操作。';
      $('statusMessage').textContent='正在同步共享地图状态';
    }
    const request=(async()=>{
      let retry=false,snapshotApplied=false;
      try {
        const snapshot=await post('/api/session/scan-state',{session_id:world.session_id},15000);
        if(world!==state.world||revision!==state.worldRevision||controlGeneration!==state.controlGeneration||requestRevision!==state.sessionSnapshotRevision)return false;
        if(staleSessionSnapshot(snapshot)) {
          const newest=Math.max(state.serverSnapshotVersion??-1,state.pendingServerSnapshotRevision??-1);
          if(state.pendingServerSnapshotRevision===null)state.pendingServerSnapshotRevision=newest;
          if(state.pendingServerSnapshotRetryRevision!==newest) {
            state.pendingServerSnapshotRetryRevision=newest;
            retry=true;
          }
          return false;
        }
        if(!applySessionSnapshot(snapshot,world))return false;
        snapshotApplied=true;
        state.sessionSnapshotRequired=false;
        if(!state.controlBusy&&!state.controlToken) {
          $('flightState').textContent='待命';
          $('flightNote').textContent=state.scanRequestCapacityReached
            ?'共享地图状态已同步；当前地图扫描请求额度已用完，可查看已扫描数据；请重置地图或生成随机地图后继续扫描。'
            :state.coverage>=1
              ?'共享地图状态已同步；扫描覆盖已完成，可查看地图或规划路线；重置地图或生成随机地图后可重新扫描。'
              :'共享地图状态已同步；点击“开始自适应扫描”重新接管后可继续操作。';
          $('statusMessage').textContent='共享地图状态已同步';
        }
        publishBlockedEndpointWarning();
        return true;
      } catch {
        retry=world===state.world&&revision===state.worldRevision&&controlGeneration!==state.controlGeneration&&!state.pageSessionReleased;
        if(world===state.world&&revision===state.worldRevision&&!state.pageSessionReleased) {
          $('flightState').textContent='地图状态待同步';
          $('flightNote').textContent='共享地图状态尚未同步；写操作已暂时锁定，服务恢复后会重新同步。';
          $('statusMessage').textContent='共享地图状态待同步';
        }
        return false;
      } finally {
        if(state.sessionSnapshotPromise===request)state.sessionSnapshotPromise=null;
        updateControlStates();
        if(retry&&state.sessionSnapshotRequired)void reconcileSessionSnapshot(world,revision);
        else if(snapshotApplied&&Number.isInteger(state.pendingServerSnapshotRevision)&&
                Number.isInteger(state.serverSnapshotVersion)&&state.serverSnapshotVersion<state.pendingServerSnapshotRevision&&
                state.pendingServerSnapshotRetryRevision!==state.pendingServerSnapshotRevision) {
          state.pendingServerSnapshotRetryRevision=state.pendingServerSnapshotRevision;
          void reconcileSessionSnapshot(world,revision);
        }
      }
    })();
    state.sessionSnapshotPromise=request;
    updateControlStates();
    return request;
  }
  function isExpiredSessionError(error) {
    return error.httpResponse&&error.message.includes('地图会话不存在或已过期');
  }
  function markScanRequestCapacityReached(confirmedPosition=null,confirmedFlightDistance=null) {
    state.scanRequestCapacityReached=true;
    const movement=state.movement;
    discardUnconfirmedMovementDistance(movement);
    if(Number.isFinite(movement?.serverConfirmedFlightDistance))state.flightDistance=movement.serverConfirmedFlightDistance;
    else if(Number.isFinite(confirmedFlightDistance))state.flightDistance=confirmedFlightDistance;
    $('flightDistanceText').textContent=`${(state.flightDistance*(state.world?.cell_size_m||100)/1000).toFixed(2)} km`;
    const anchor=movement?.world===state.world?movement.scanCursor:confirmedPosition;
    if(pointInWorld(anchor,state.world)) {
      const previous=state.lastDrone||state.scanTrail[state.scanTrail.length-1];
      state.dronePosition={...anchor};
      if(previous&&(previous.x!==anchor.x||previous.y!==anchor.y))state.previousDronePosition={...previous};
      const last=state.scanTrail[state.scanTrail.length-1];
      if(!last||last.x!==anchor.x||last.y!==anchor.y)state.scanTrail.push({...anchor});
      state.lastDrone={...anchor};
    }
    stopFlight();
    draw();
    $('flightState').textContent='需要新地图';
    $('flightNote').textContent='当前地图的扫描请求额度已用完；已确认数据仍保留，请重置地图或生成随机地图后继续。';
    $('statusMessage').textContent='当前地图无法继续接收新扫描，请重置或生成新地图';
    updateControlStates();
  }
  function isUnsupportedScanApiError(error) {
    return error.httpStatus===404&&error.message==='未知接口';
  }
  function unsupportedScanApiNote() {
    return '当前仿真服务版本缺少扫描接口（HTTP 404）；本段扫描请求已保留。重启为当前版本后，点击“开始自适应扫描”可安全恢复。';
  }
  function scanRequestId() { return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`; }
  function syncMissionPoints(world=state.world,worldRevision=state.worldRevision) {
    if(!world||worldRevision!==state.worldRevision||world!==state.world||state.pageSessionReleased||state.sessionSnapshotRequired)return Promise.resolve(false);
    if(state.missionSyncedRevision===state.endpointRevision)return Promise.resolve(true);
    const update=state.missionUpdateQueue.catch(()=>false).then(async()=>{
      if(worldRevision!==state.worldRevision||world!==state.world)return false;
      const endpointRevision=state.endpointRevision,start={...world.start},goal={...world.goal};
      try {
        const payload={session_id:world.session_id,start,goal},movement=state.movement,pending=state.pendingScan?.world===world?state.pendingScan:null;
        const segments=pending?.segments?.length?pending.segments:pending?.payload?[pending.payload]:[];
        const pendingLegId=segments.length&&segments.every(segment=>segment.leg_id===segments[0].leg_id)?segments[0].leg_id:null;
        const keepLegId=movement?.world===world&&movement.revision===worldRevision&&movement.legId
          ?movement.legId:typeof pendingLegId==='string'&&pendingLegId?pendingLegId:null;
        if(keepLegId)payload.keep_leg_id=keepLegId;
        const configured=await post('/api/mission/points',payload,15000);
        if(worldRevision===state.worldRevision&&world===state.world&&endpointRevision===state.endpointRevision&&
           !pointInWorld(configured?.drone_position,world))throw new Error('服务端返回的无人机位置无效');
        if(worldRevision===state.worldRevision&&world===state.world&&endpointRevision===state.endpointRevision&&
           !state.movement&&!state.pendingScan) {
          state.dronePosition={...configured.drone_position};
          if(state.dataRevision===0&&state.flightDistance===0) {
            state.scanTrail=[{...configured.drone_position}];
            state.previousDronePosition=null; state.lastDrone=null;
          }
          draw();
        }
      } catch(e) {
        if(worldRevision===state.worldRevision&&world===state.world&&endpointRevision===state.endpointRevision) {
          if(e.code==='control_conflict')return false;
          $('statusMessage').textContent=`起终点同步失败；扫描开始前会重试：${e.message}`;
        }
        return false;
      }
      if(worldRevision!==state.worldRevision||world!==state.world||endpointRevision!==state.endpointRevision)return false;
      state.missionSyncedRevision=endpointRevision;
      return true;
    });
    state.missionUpdateQueue=update;
    return update;
  }
  function markSessionExpired() {
    const lostPendingScan=Boolean(state.pendingScan);
    const movement=state.movement?.world===state.world&&state.movement.revision===state.worldRevision?state.movement:null;
    if(movement) {
      discardUnconfirmedMovementDistance(movement);
      if(Number.isFinite(movement.serverConfirmedFlightDistance))state.flightDistance=movement.serverConfirmedFlightDistance;
      if(pointInWorld(movement.scanCursor,state.world)) {
        const anchor=movement.scanCursor,previous=state.lastDrone||state.scanTrail[state.scanTrail.length-1];
        state.dronePosition={...anchor};
        if(previous&&(previous.x!==anchor.x||previous.y!==anchor.y))state.previousDronePosition={...previous};
        const last=state.scanTrail[state.scanTrail.length-1];
        if(!last||last.x!==anchor.x||last.y!==anchor.y)state.scanTrail.push({...anchor});
        state.lastDrone={...anchor};
      }
      $('flightDistanceText').textContent=`${(state.flightDistance*(state.world.cell_size_m||100)/1000).toFixed(2)} km`;
    }
    state.sessionExpired=true;
    state.worldSessionUnconfirmed=false;
    state.planAfterScan=false;
    state.pendingScan=null;
    state.retryPlanPending=false;
    $('flightState').textContent='地图会话已过期';
    $('flightNote').textContent=lostPendingScan?'当前地图会话已失效，未同步的扫描航段无法恢复；点击“随机地图”或“重置地图”重新生成。':'当前地图会话已失效，无法继续扫描或规划；点击“随机地图”或“重置地图”重新生成。';
    $('statusMessage').textContent='地图会话已过期，请重新生成地图';
    stopFlight();
    updateControlStates();
  }
  async function sense(from,p,world,revision,legId) {
    const requestId=scanRequestId();
    const payload={session_id:world.session_id,start:from,end:p,radius:4,leg_id:legId,request_id:requestId};
    const pending={world,payload,segments:[payload],confirmedFlightDistance:state.movement?.world===world?state.movement.serverConfirmedFlightDistance:state.flightDistance};
    state.pendingScan=pending;
    updateControlStates();
    const result=await post('/api/scan/path',pending.payload,15000);
    if(world!==state.world||revision!==state.worldRevision)return false;
    if(result.reconcile_snapshot===true) {
      const snapshot=await post('/api/session/scan-state',{session_id:world.session_id},15000);
      if(world!==state.world||revision!==state.worldRevision||!applySessionSnapshot(snapshot,world))return false;
      updateControlStates();return true;
    }
    if(!applySenseResult(result,world))return false;
    if(result.drone_position&&!(state.movement?.world===world&&state.movement?.revision===revision))state.dronePosition={...result.drone_position};
    const movement=state.movement;
    if(movement?.world===world&&movement.revision===revision&&Number.isFinite(result.flight_distance)&&result.flight_distance>=0)movement.serverConfirmedFlightDistance=result.flight_distance;
    if(state.pendingScan===pending){const segments=(pending.segments||[pending.payload]).filter(segment=>segment.request_id!==payload.request_id);if(segments.length){pending.segments=segments;pending.payload=segments[0];}else state.pendingScan=null;}
    updateControlStates();
    return true;
  }
  async function retryPendingScan() {
    const pending=state.pendingScan;
    if(!pending)return true;
    if(pending.world!==state.world){state.pendingScan=null;return true;}
    const revision=state.worldRevision;
    const isCurrent=()=>pending.world===state.world&&revision===state.worldRevision&&!state.sessionExpired&&state.pendingScan===pending;
    const dataRevision=state.dataRevision;
    let confirmedFlightDistance=null;
    let confirmedPosition=null;
    try {
      for(const payload of pending.segments||[pending.payload]) {
        const result=await post('/api/scan/path',payload,15000);
        if(!isCurrent())return false;
        if(result.reconcile_snapshot===true) {
          const snapshot=await post('/api/session/scan-state',{session_id:pending.world.session_id},15000);
          if(!isCurrent()||!applySessionSnapshot(snapshot,pending.world))return false;
          if(state.pendingScan===pending)state.pendingScan=null;
          refreshCoverage();draw();announceScanRecovery(pending.world);return true;
        }
        if(!applySenseResult(result,pending.world))return false;
        if(result.drone_position)state.dronePosition={...result.drone_position};
        confirmedPosition=pointInWorld(result.drone_position,pending.world)?{...result.drone_position}:{...payload.end};
        if(Number.isFinite(result.flight_distance)&&result.flight_distance>=0)confirmedFlightDistance=result.flight_distance;
      }
      if(state.pendingScan===pending)state.pendingScan=null;
      if(confirmedFlightDistance!==null) {
        state.flightDistance=confirmedFlightDistance;
        $('flightDistanceText').textContent=`${(state.flightDistance*(state.world.cell_size_m||100)/1000).toFixed(2)} km`;
      }
      refreshCoverage(); draw(); consumeDeferredPlan(pending.world);announceScanRecovery(pending.world);
      return true;
    } catch(e) {
      if(!isCurrent())return false;
      const partiallySynced=state.dataRevision!==dataRevision;
      if(confirmedFlightDistance!==null) {
        state.flightDistance=confirmedFlightDistance;
        $('flightDistanceText').textContent=`${(state.flightDistance*(state.world.cell_size_m||100)/1000).toFixed(2)} km`;
      }
      if(partiallySynced){refreshCoverage();draw();}
      if(e.code==='scan_request_capacity'){
        const first=pending.segments?.[0]||pending.payload;
        markScanRequestCapacityReached(confirmedPosition||first.start||null,confirmedFlightDistance??pending.confirmedFlightDistance);
        return false;
      }
      if(e.code==='control_conflict')return false;
      if(isExpiredSessionError(e)) { markSessionExpired(); return false; }
      const unsupported=isUnsupportedScanApiError(e);
      $('flightState').textContent=unsupported?'服务版本不兼容':partiallySynced?'部分扫描已同步':'扫描数据待同步';
      $('flightNote').textContent=unsupported?unsupportedScanApiNote():partiallySynced?`前序扫描段已确认并更新地图；剩余航段未确认：${e.message}。服务恢复后再次点击“开始自适应扫描”可安全重试。`:`上一段扫描结果尚未确认：${e.message}。服务恢复后再次点击“开始自适应扫描”可安全重试并同步数据。`;
      $('statusMessage').textContent=unsupported?'仿真服务版本过旧；扫描请求已保留':partiallySynced?'部分扫描结果已同步；未确认航段已保留，可恢复后重试':'上次扫描结果尚未同步；请检查服务后重试';
      return false;
    }
  }
  async function flushMovementScan(movement) {
    if(movement.scanPending)return;
    const same=(a,b)=>a.x===b.x&&a.y===b.y;
    if(same(movement.scanCursor,movement.scanTarget)) {
      if(movement.arrived&&state.movement===movement)finishMovement(true);
      return;
    }
    movement.scanPending=true;
    const from={...movement.scanCursor},to={...movement.scanTarget};
    const targetDistance=movement.scanDistanceTarget;
    try {
      if(await sense(from,to,movement.world,movement.revision,movement.legId)) {
        movement.scanCursor=to;movement.confirmedDistance=Math.max(movement.confirmedDistance,targetDistance);
        refreshCoverage(); draw(); consumeDeferredPlan(movement.world);
      }
    } catch(e) {
      if(movement.world===state.world&&movement.revision===state.worldRevision) {
        if(e.code==='scan_request_capacity'){markScanRequestCapacityReached();return;}
        if(e.code==='control_conflict')return;
        if(isExpiredSessionError(e)){markSessionExpired();return;}
        const pending=state.pendingScan, current={x:Math.round(state.dronePosition.x),y:Math.round(state.dronePosition.y)};
        if(pending?.world===movement.world) {
          const segments=pending.segments||[pending.payload], last=segments[segments.length-1].end;
          if(last.x!==current.x||last.y!==current.y) {
            segments.push({...pending.payload,start:{...last},end:current,
              request_id:`${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`});
            pending.segments=segments;
          }
        }
        const unsupported=isUnsupportedScanApiError(e);
        $('flightState').textContent=unsupported?'服务版本不兼容':'扫描数据待同步';
        $('flightNote').textContent=unsupported?unsupportedScanApiNote():`传感器读取失败：${e.message}。本段请求已保留；服务恢复后点击“开始自适应扫描”重试并同步数据。`;
        $('statusMessage').textContent=unsupported?'仿真服务版本过旧；扫描请求已保留':'航段扫描失败，等待结果同步'; stopFlight();
      }
    } finally { movement.scanPending=false; }
    if(state.movement!==movement)return;
    if(!same(movement.scanCursor,movement.scanTarget)) { void flushMovementScan(movement); return; }
    if(movement.arrived)finishMovement(true);
  }
  function preserveMovementScanTail() {
    const movement=state.movement;
    if(!movement||movement.world!==state.world||!state.dronePosition)return;
    const pending=state.pendingScan;
    if(pending&&pending.world!==movement.world)return;
    const segments=pending?(pending.segments||[pending.payload]):[];
    const last=segments.length?segments[segments.length-1].end:movement.scanCursor;
    const end={x:Math.round(state.dronePosition.x),y:Math.round(state.dronePosition.y)};
    if(last.x===end.x&&last.y===end.y)return;
    const payload=pending?{...pending.payload,start:{...last},end,request_id:scanRequestId()}:
      {session_id:movement.world.session_id,start:{...last},end,radius:4,leg_id:movement.legId,request_id:scanRequestId()};
    if(pending){segments.push(payload);pending.segments=segments;}
    else state.pendingScan={world:movement.world,payload,segments:[payload],confirmedFlightDistance:movement.serverConfirmedFlightDistance};
  }
  function endpointsKnown() {
    return state.observed[state.world.start.y]?.[state.world.start.x]&&state.observed[state.world.goal.y]?.[state.world.goal.x];
  }
  function knownBlockedEndpoints() {
    return ['start','goal'].filter(which=>{
      const point=state.world?.[which];
      return point&&state.observed[point.y]?.[point.x]&&isHardBlockedCell(state.world.layers,point.y,point.x);
    });
  }
  function blockedEndpointMessage(endpoints) {
    const labels=endpoints.map(which=>`${which==='start'?'起点':'目标点'}（X ${state.world[which].x}，Y ${state.world[which].y}）`);
    const names=endpoints.map(which=>which==='start'?'起点':'目标点').join('和');
    return `${labels.join('、')}已扫描，当前代价场判定为禁行区域，请调整${names}后再规划。`;
  }
  function publishBlockedEndpointWarning(endpoints=knownBlockedEndpoints()) {
    if(!endpoints.length)return false;
    const message=blockedEndpointMessage(endpoints);
    $('planningBadge').textContent='端点禁行';
    if($('statusMessage').textContent!==message)$('statusMessage').textContent=message;
    $('flightNote').textContent=`${message}当前端点无法生成可行路线。`;
    return true;
  }
  function consumeDeferredPlan(world) {
    if(!state.planAfterScan||state.pendingScan||state.plannerReady===false)return;
    if(world!==state.world||state.coverage<.20||!endpointsKnown())return;
    state.planAfterScan=false;
    state.scanSincePlan=Math.max(state.scanSincePlan,Math.floor(state.coverage/.20));
    void planRoute();
  }
  function refreshCoverage() {
    const total=state.world.width*state.world.height, seen=state.observed.reduce((n,row)=>n+row.filter(Boolean).length,0); state.coverage=seen/total;
    const elevations=[];
    for(let y=0;y<state.world.height;y++) for(let x=0;x<state.world.width;x++) if(state.observed[y][x]) elevations.push(state.world.layers.elevation[y][x]);
    state.elevationRange=elevations.length?[Math.min(...elevations),Math.max(...elevations)]:[0,1];
    const coveragePercent=state.coverage>=1?100:Math.min(99.9,Math.round(state.coverage*1000)/10),coverageLabel=`${Number.isInteger(coveragePercent)?coveragePercent:coveragePercent.toFixed(1)}%`;
    $('coverageText').textContent=coverageLabel; $('coverageText').setAttribute('aria-valuenow',String(coveragePercent)); $('coverageBar').style.width=`${state.coverage*100}%`; $('cellsMetric').textContent=seen.toLocaleString();
    const milestone=state.coverage>=1?100:Math.floor(state.coverage*10)*10;
    if(state.flying&&milestone>state.announcedCoverageMilestone&&milestone<100){
      const position=state.dronePosition||state.world.start;
      const distanceKm=(state.flightDistance*(state.world.cell_size_m||100)/1000).toFixed(2);
      $('coverageAnnouncement').textContent=`扫描覆盖率 ${milestone}%；无人机位置 X ${Math.round(position.x)}、Y ${Math.round(position.y)}；累计航程 ${distanceKm} 公里`;
      state.announcedCoverageMilestone=milestone;
    }
    if(milestone===100)state.announcedCoverageMilestone=100;
    state.guidanceWeight=clamp(.75+state.coverage*1.25,.75,2.0); $('guidanceLabel').textContent=state.guidanceWeight.toFixed(2);
    updateControlStates();
  }
  function setWorldLoading(loading) {
    if(loading){const active=document.activeElement;state.worldLoadingFocus=['randomBtn','resetBtn'].includes(active?.id)?active:null;}
    state.worldLoading=loading;
    updateControlStates();
    if(!loading){const focus=state.worldLoadingFocus;state.worldLoadingFocus=null;if(focus?.isConnected&&(document.activeElement===focus||document.activeElement===document.body)){if(!focus.disabled)focus.focus();else if($('randomBtn').isConnected&&!$('randomBtn').disabled)$('randomBtn').focus();}}
  }
  async function makeWorld(seed,retryPayload=null) {
    preserveMovementScanTail();
    if(state.planning&&state.world&&state.coverage>=.20&&!state.route.length)state.retryPlanPending=true;
    const intentRevision=++state.worldRevision, previousWorld=state.world; state.planRevision++; state.planning=false; state.planQueued=false;
    state.worldLoadFailed=false;
    $('planningBadge').textContent='地图生成中';
    const payload=retryPayload||{seed,width:144,height:108,request_id:window.crypto?.randomUUID?.()||`${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`};
    if(!retryPayload){const replaceSessionId=previousWorld?.session_id||startupReplacementSessionId;if(replaceSessionId)payload.replace_session_id=replaceSessionId;}
    state.pendingWorldCreate=payload;
    persistPendingWorld(payload);
    setWorldLoading(true); stopFlight(); $('flightState').textContent='生成地图中'; $('flightNote').textContent=retryPayload?'正在恢复上次未确认的地图创建请求；收到结果后将更新地图。':state.world?'新地图生成中；当前任务将在新地图结果确认后切换。':'正在生成地图，完成后即可开始扫描。'; $('mapEmpty').classList.remove('hidden','failed','idle'); $('mapStateTitle').textContent='正在生成仿真地形';
    $('statusMessage').textContent=retryPayload?'正在恢复上次未确认的地图创建…':'生成随机地形与风险图层…';
    try {
      const world=await postWorld(payload,intentRevision);
      if(intentRevision!==state.worldRevision)return;
      if(!world||typeof world!=='object'||Array.isArray(world))throw invalidWorldDataError(world);
      if(world.api_protocol!==API_PROTOCOL){const error=new Error('仿真服务接口版本不匹配，请重启仿真服务后刷新页面');error.protocolMismatch=true;error.sessionId=world.session_id;throw error;}
      if(!validWorldResponse(world))throw invalidWorldDataError(world);
      const snapshot=world.request_replayed===true?await post('/api/session/scan-state',{session_id:world.session_id},15000):null;
      if(intentRevision!==state.worldRevision)return;
      if(snapshot&&!validSessionSnapshot(snapshot,world))throw invalidWorldDataError(world);
      if(snapshot){world.start={...snapshot.mission_start};world.goal={...snapshot.mission_goal};}
      world.layers=Object.fromEntries(world.layer_names.map(name=>[name,matrix(world.height,world.width,null)]));
      rememberSessionId(world.session_id);clearPersistedPendingWorld();startupReplacementSessionId=null;
      state.world=world; state.worldRequestSeed=payload.seed; state.serverInstance=world.instance_id||state.serverInstance; state.serverSnapshotVersion=Number.isInteger(snapshot?.snapshot_revision)?snapshot.snapshot_revision:0; state.pendingServerSnapshotRevision=null; state.pendingServerSnapshotRetryRevision=null; state.controlGeneration++;state.sessionSnapshotRevision++;state.sessionSnapshotRequired=false;state.sessionSnapshotPromise=null;state.controlToken=null;state.controlEpoch=0;state.controlOwner=null;state.controlBusy=false;state.controlLost=false;state.pageSessionReleased=false;state.scanRequestCapacityReached=false;state.startingFlight=false;state.baseSignature=''; state.pendingScan=null; state.retryPlanPending=false; state.planAfterScan=false; state.plannedDataRevision=-1; state.keyboardMapUsed=false; state.sessionExpired=false; state.worldSessionUnconfirmed=false; state.pendingWorldCreate=null; state.missionSyncedRevision=state.endpointRevision; state.missionUpdateQueue=Promise.resolve(true); state.mapZoom=1; applyMapCanvasSize(); $('mapViewport').scrollLeft=0; $('mapViewport').scrollTop=0;
      const {width,height}=state.world; state.observed=matrix(height,width,false); state.confidence=matrix(height,width,0); state.samples=matrix(height,width,0); state.route=[]; state.scanPath=[]; state.dronePosition=snapshot?{...snapshot.drone_position}:{...world.start}; state.previousDronePosition=snapshot?.previous_drone_position?{...snapshot.previous_drone_position}:null; state.lastDrone=snapshot?{...snapshot.drone_position}:null; state.scanTrail=[{...state.dronePosition}]; state.flightDistance=Number.isFinite(snapshot?.flight_distance)?snapshot.flight_distance:0; state.scanSincePlan=0; state.planCount=0; state.announcedCoverageMilestone=0; $('coverageAnnouncement').textContent=''; $('scanAnnouncement').textContent=''; state.costGridCache=null; state.layerRangeCache={};state.layerLegendSignature=''; state.coverage=0; state.elevationRange=[0,1]; $('flightDistanceText').textContent=`${(state.flightDistance*(state.world.cell_size_m||100)/1000).toFixed(2)} km`;
      $('seedLabel').textContent=state.world.map_id; $('resolution').textContent=`${state.world.cell_size_m||100} m`; $('dimensions').textContent=`${width} × ${height}`; $('mapEmpty').classList.add('hidden'); $('mapEmpty').classList.remove('failed'); $('planningBadge').textContent='尚未规划'; $('distanceMetric').textContent='—'; $('costMetric').textContent='—'; $('searchMetric').textContent='—';
      const select=$('layerSelect'), selected=select.value; select.replaceChildren(new Option('综合代价','cost'),new Option('归一化高程','elevation'),new Option('观测置信度','confidence'));
      Object.keys(state.world.layers).filter(k=>!['cost','elevation'].includes(k)).forEach(k=>select.add(new Option(layerLabels[k]||k,k)));
      select.value=(state.world.layers[selected]||selected==='cost'||selected==='confidence')?selected:'elevation';
      if(snapshot){applySenseResult(snapshot,world);state.route=Array.isArray(snapshot.route)?snapshot.route.map(point=>({...point})):[];state.scanSincePlan=Math.floor(state.coverage/.20);state.plannedDataRevision=state.dataRevision;if(typeof snapshot.scan_request_capacity_reached==='boolean')state.scanRequestCapacityReached=snapshot.scan_request_capacity_reached;}
      state.scanPath=scanPattern(); state.keyboardPoint={...world.start}; updatePoints(); refreshCoverage(); setWorldLoading(false); draw(); $('flightState').textContent=state.scanRequestCapacityReached?'需要新地图':state.coverage>=1?'扫描完成':'待命'; $('planningBadge').textContent=state.plannerReady===false?'规划器不可用':state.route.length?'已恢复路线':'尚未规划';
      if(snapshot&&state.route.length>0){const routeDistance=state.route.slice(1).reduce((sum,point,index)=>sum+Math.hypot(point.x-state.route[index].x,point.y-state.route[index].y),0);$('distanceMetric').textContent=((routeDistance*(world.cell_size_m||100))/1000).toFixed(2);$('searchMetric').textContent=state.route.length.toLocaleString();}
      $('flightNote').textContent=state.scanRequestCapacityReached?'已恢复已确认数据；当前地图扫描请求额度已用完，请重置地图或生成随机地图后继续。':state.coverage>=1?'已恢复完整扫描地图；当前覆盖率 100%，无需继续扫描。':state.plannerReady===false?`地图扫描可继续；${plannerUnavailableMessage()}`:snapshot?`已恢复 ${$('cellsMetric').textContent} 个已扫描格、任务端点和无人机位置；可继续执行扫描。`:'启动后边飞行边扫描；覆盖率达到 20% 后自动规划，并沿当前路线继续探测。';
      $('statusMessage').textContent=state.scanRequestCapacityReached?'扫描会话已恢复，但本地图扫描请求额度已用完，请重置或生成随机地图':state.coverage>=1?'扫描会话已恢复；当前覆盖率 100%':snapshot?'已恢复扫描会话，可继续执行任务':'陌生地形已生成，等待扫描';
    } catch(e) {
      if(intentRevision!==state.worldRevision||previousWorld!==state.world)return;
      if(e.protocolMismatch) {
        if(e.preservePendingWorldCreate) {
          state.pendingWorldCreate=payload;persistPendingWorld(payload);
          await releaseRejectedWorldSession(e);
          if(intentRevision!==state.worldRevision||previousWorld!==state.world)return;
        } else {
          rememberSessionId(e.sessionId);startupReplacementSessionId=e.sessionId||startupReplacementSessionId;
          state.pendingWorldCreate=null;clearPersistedPendingWorld();
        }
        showProtocolMismatch(e.message);return;
      }
      const capacityRejected=isWorldCapacityRejection(e);
      if(e.transportError||(e.httpStatus>=500&&!capacityRejected))state.pendingWorldCreate=payload;
      else if(state.pendingWorldCreate?.request_id===payload.request_id){state.pendingWorldCreate=null;clearPersistedPendingWorld();}
      if(capacityRejected){state.pendingWorldCreate=payload;persistPendingWorld(payload);}
      const verification=previousWorld?await verifyWorldSession(previousWorld,intentRevision):{status:'unknown'};
      if(intentRevision!==state.worldRevision||previousWorld!==state.world||verification.status==='stale')return;
      if(verification.status==='incompatible'){showProtocolMismatch();return;}
      state.worldLoadFailed=true;
      if(verification.status==='valid'){
        state.sessionExpired=false;
        state.worldSessionUnconfirmed=state.pendingWorldCreate?.request_id===payload.request_id&&!capacityRejected;
      } else if(verification.status==='invalid') state.worldSessionUnconfirmed=false;
      else state.worldSessionUnconfirmed=!capacityRejected||Boolean(previousWorld);
      setWorldLoading(false);
      if(state.world){
        $('mapEmpty').classList.add('hidden');
        $('planningBadge').textContent=state.route.length?'旧地图路线保留':'尚未规划';
        if(verification.status==='invalid')markSessionExpired();
        const canResume=state.pendingWorldCreate?.request_id===payload.request_id;
        const requestUnconfirmed=canResume&&!capacityRejected;
        refreshCoverage(); draw();
        let flightState,flightNote,statusMessage;
        if(requestUnconfirmed) {
          flightState='结果未确认';
          flightNote=verification.status==='valid'?`新地图创建结果仍未确认；旧地图会话已核验有效。${state.connected===false?'服务恢复后点击“随机地图”重放相同请求。':'点击“随机地图”将重放相同请求。'}`:`新地图创建结果未确认；旧地图会话状态无法核验，核验前不能飞行或规划。${state.connected===false?'服务恢复后点击“随机地图”重放相同请求。':'点击“随机地图”将重放相同请求。'}`;
          statusMessage=`地图创建结果未确认：${e.message}`;
        } else if(verification.status==='unknown') {
          flightState='地图状态待同步';
          flightNote=`新地图创建失败：${e.message}；旧地图会话状态无法核验，扫描、规划和端点编辑暂时锁定。${state.connected===false?'服务恢复后重新生成地图。':'同步完成后可继续操作。'}`;
          statusMessage=`新地图创建已失败，但旧地图状态待核验：${e.message}`;
        } else if(state.sessionExpired) {
          flightState='地图会话已过期';
          flightNote=`旧地图会话已失效；${canResume?'点击“随机地图”可恢复未确认请求。':'请点击“随机地图”重新生成。'}`;
          statusMessage=`新地图创建失败，旧地图会话已失效：${e.message}`;
        } else if(state.pageSessionReleased) {
          flightState='当前页面只读';
          flightNote=capacityRejected?'新地图创建因会话容量被拒绝；本页已释放旧地图控制权，当前只能查看。关闭其他地图标签页后点击“随机地图”重试。':`新地图创建失败：${e.message}；本页已释放旧地图控制权，当前只能查看。修复问题后点击“随机地图”重试。`;
          statusMessage=`当前页面已释放旧地图控制权；新地图创建${capacityRejected?'因会话容量被拒绝':'失败'}：${e.message}`;
        } else if(state.sessionSnapshotRequired) {
          flightState='地图状态待同步';
          flightNote=`新地图创建失败：${e.message}；共享地图状态仍在同步，旧地图的扫描、规划和端点编辑暂时锁定；随机地图和重置地图仍可用。`;
          statusMessage=`共享地图状态待同步；新地图创建失败：${e.message}`;
        } else if(state.scanRequestCapacityReached) {
          flightState='需要新地图';
          flightNote=capacityRejected?`新地图创建因会话容量已满被拒绝；旧地图会话有效，但扫描请求额度已耗尽。关闭其他地图标签页后，点击“随机地图”重试。`:`新地图生成失败：${e.message}；旧地图会话有效，但扫描请求额度已耗尽。检查问题后重试生成地图。`;
          statusMessage=`新地图创建${capacityRejected?'被会话容量拒绝':'失败'}；旧地图扫描请求额度已耗尽：${e.message}`;
        } else if(state.coverage>=1&&!state.pendingScan) {
          flightState='扫描完成';
          flightNote=capacityRejected?`旧地图扫描覆盖已完成；新地图创建因会话容量已满被拒绝。关闭其他地图标签页后点击“随机地图”重试。`:`旧地图扫描覆盖已完成；新地图生成失败：${e.message}。检查问题后重试。`;
          statusMessage=`旧地图扫描已完成；新地图创建${capacityRejected?'被会话容量拒绝':'失败'}：${e.message}`;
        } else if(capacityRejected&&verification.status==='valid') {
          flightState=state.controlBusy?'其他页面控制中':'待命';
          flightNote=state.controlBusy?'另一页面正在控制旧地图；控制权释放后可继续操作。':state.plannerReady===false?`旧地图会话有效，可以继续扫描；${plannerUnavailableMessage()} 关闭其他地图标签页后，点击“随机地图”重试。`:`新地图创建因会话容量已满被拒绝；旧地图会话有效，可以继续扫描和规划。关闭其他地图标签页后，点击“随机地图”会重试同一请求。`;
          statusMessage=`新地图创建被会话容量拒绝；旧地图可继续使用：${e.message}`;
        } else {
          flightState='待命';
          flightNote=requestUnconfirmed?`旧地图会话已核验有效，但新地图请求仍未确认；请点击“随机地图”用同一请求恢复结果。`:`新地图生成失败，已核验旧地图会话有效；当前地图（编号 ${state.world.map_id}）可以继续。`;
          statusMessage=requestUnconfirmed?`新地图创建结果仍未确认，旧地图会话有效；点击“随机地图”将重放同一请求：${e.message}`:`新地图生成失败，旧地图会话已核验有效：${e.message}`;
        }
        $('flightState').textContent=flightState;
        $('flightNote').textContent=flightNote;
        $('statusMessage').textContent=statusMessage;
      } else {
        const requestUnconfirmed=state.pendingWorldCreate?.request_id===payload.request_id&&!capacityRejected;
        state.worldSessionUnconfirmed=requestUnconfirmed;
        const retryMessage=capacityRejected?`关闭其他地图标签页后，${state.connected===false?'服务恢复后点击“随机地图”重试同一请求。':'点击“随机地图”重放同一请求。'}`:requestUnconfirmed?(state.connected===false?'服务恢复后点击“随机地图”重放相同请求。':'点击“随机地图”将重放相同请求。'):'修正问题后点击“随机地图”重新生成。';
        const title=capacityRejected?'地图会话容量已满':requestUnconfirmed?'地图结果未确认':'地图生成失败';
        $('planningBadge').textContent=title;
        $('flightState').textContent=capacityRejected?'会话容量已满':title;
        $('flightNote').textContent=capacityRejected?`地图会话容量已满，当前没有可用地图；${retryMessage}`:requestUnconfirmed?`地图创建结果未确认，当前没有可用地图；${retryMessage}${e.message}`:`地图生成失败：${e.message}。${retryMessage}`;
        $('mapEmpty').classList.remove('hidden');$('mapEmpty').classList.add('failed');$('mapStateTitle').textContent=title;$('mapStateDetail').textContent=capacityRejected?retryMessage:requestUnconfirmed?`${retryMessage}${e.message}`:`${e.message}。${retryMessage}`;
        $('statusMessage').textContent=capacityRejected?`地图会话容量已满，创建请求未提交：${e.message}`:requestUnconfirmed?`地图创建结果未确认：${e.message}`:`地图生成失败：${e.message}`;
      }
      updateControlStates();
      const retryAfterLoad=state.retryPlanPending&&state.connected!==false&&state.world&&state.coverage>=.20&&!state.route.length;
      state.retryPlanPending=false;
      if(retryAfterLoad)planRoute();
    }
  }
  async function initializeSimulator() {
    const reloadState=restoreReloadState();
    startupReplacementSessionId=reloadState.sessionId;
    state.pendingWorldCreate=reloadState.pendingWorldCreate;
    await checkServer();
    if(state.connected){const pending=state.pendingWorldCreate;await makeWorld(pending?.seed??Math.floor(Math.random()*900000)+100000,pending);}
  }
  function scanPattern() {
    const {width,height}=state.world, points=[],s=state.world.start;
    // Start on the nearest raster row, then alternate direction for a continuous S sweep.
    const rows=Array.from({length:Math.ceil((height-1)/3)+1},(_,i)=>Math.min(height-1,i*3));
    if(rows[rows.length-1]!==height-1) rows.push(height-1);
    if(s.y>height/2) rows.reverse();
    let reverse=s.x>width/2;
    for(const y of rows) { const row=[]; if(!reverse) for(let x=1;x<width-1;x+=2)row.push({x,y}); else for(let x=width-2;x>0;x-=2)row.push({x,y}); points.push(...row);reverse=!reverse; }
    return points;
  }
  async function planRoute() {
    if(!state.world||state.worldLoading||state.startingFlight||state.controlBusy||state.pageSessionReleased||state.sessionSnapshotRequired)return;
    if(state.plannerReady===false){showPlannerUnavailable();return;}
    if(state.worldSessionUnconfirmed){$('statusMessage').textContent='地图会话状态未确认；核验完成前不能规划';return;}
    if(state.sessionExpired){$('statusMessage').textContent='地图会话已过期，请点击“随机地图”或“重置地图”重新生成';return;}
    if(state.pendingScan){state.planAfterScan=true;$('statusMessage').textContent='上段扫描结果尚未同步；同步后将按当前起终点规划';return;}
    if(publishBlockedEndpointWarning()) {
      updateControlStates();
      return;
    }
    if(state.connected===false){$('statusMessage').textContent='仿真服务离线；恢复后将继续规划';return;}
    if(state.coverage<.20){$('statusMessage').textContent='请先扫描地图，覆盖率达到 20% 后再规划';return;}
    if(state.planning){state.planQueued=true;return;}
    const world=state.world, worldRevision=state.worldRevision, endpointRevision=state.endpointRevision;
    const missionSynced=await syncMissionPoints(world,worldRevision);
    if(!missionSynced||world!==state.world||worldRevision!==state.worldRevision||endpointRevision!==state.endpointRevision)return;
    if(state.planning){state.planQueued=true;return;}
    const requestId=++state.planRevision;
    state.planning=true;
    state.plannedDataRevision=state.dataRevision;
    $('planBtn').disabled=true; $('statusMessage').textContent='基于当前已扫描数据计算低代价路线…'; $('planningBadge').textContent='规划中';
    let responseReceived=false;
    try {
      const result=await post('/api/plan',{session_id:world.session_id,start:{...world.start},goal:{...world.goal}},30000);
      responseReceived=true;
      if(requestId!==state.planRevision||world!==state.world)return;
      state.route=result.path||[];
      if(result.waiting_for_scan||!state.route.length) {
        $('distanceMetric').textContent='—'; $('costMetric').textContent='—'; $('searchMetric').textContent='—';
        $('planningBadge').textContent=state.coverage>=1&&!result.waiting_for_scan?'无可行路线':'等待更多扫描';
        $('statusMessage').textContent=state.paused
          ?result.waiting_for_scan?'扫描已暂停；请继续扫描，观测到起点和终点后再规划。':state.coverage>=1?'扫描已暂停；完整地图中未找到可行路线，请调整起终点。':'扫描已暂停；已观测区域暂未连通或端点位于硬约束区，继续扫描可扩展可行区域。'
          :result.waiting_for_scan?'起点或终点尚未被扫描，继续扫描后规划':state.coverage>=1?'完整地图中未找到可行路线，请调整起终点':'已观测区域暂未连通或端点位于硬约束区，继续扫描或调整端点';
        if(!state.flying||state.paused)$('flightNote').textContent=state.paused
          ?result.waiting_for_scan?'扫描已暂停；起点或终点尚未被扫描，继续扫描后再规划。':state.coverage>=1?'扫描已暂停；完整地图中当前端点间没有可行路线，请调整起终点。':'扫描已暂停；已观测区域暂未连通或端点位于硬约束区，继续扫描或调整端点。'
          :result.waiting_for_scan?'起点或终点尚未被扫描，继续扫描后再规划。':state.coverage>=1?'扫描覆盖已完成，但当前端点间没有可行路线；请调整起终点。':'已观测区域暂未连通或端点位于硬约束区；继续扫描或调整端点。';
        else $('flightNote').textContent='当前已观测区域尚未形成可行路线；无人机继续扫描并等待更多地形数据。';
        draw(); return;
      }
      state.planCount++;
      const pathLength=state.route.slice(1).reduce((sum,p,i)=>sum+Math.hypot(p.x-state.route[i].x,p.y-state.route[i].y),0);
      $('distanceMetric').textContent=((pathLength*(state.world.cell_size_m||100))/1000).toFixed(2); $('costMetric').textContent=Number(result.cost||0).toFixed(2); $('searchMetric').textContent=state.route.length.toLocaleString(); $('planningBadge').textContent=`第 ${state.planCount} 次规划`; $('statusMessage').textContent=`${result.planner||'V5 A*'} 完成 · ${state.route.length} 个路径点`;
      if(!state.flying||state.paused)$('flightNote').textContent=state.paused
        ?state.coverage>=1?'扫描已暂停；路径已更新，地图覆盖已完成。':'扫描已暂停；路径已更新，继续后将使用新路径扫描。'
        :state.coverage>=1?'路径已更新；扫描覆盖已完成。':'路径已更新；继续扫描可根据新观测动态优化。';
      else $('flightNote').textContent='扫描路径已更新；无人机将继续沿自适应路径飞行并扫描，覆盖率达到 20% 后再次规划。';
      $('startBtn').disabled=state.sessionExpired||state.flying||(state.coverage>=1&&!state.pendingScan); $('resetBtn').disabled=false; draw();
    } catch(e) { if(requestId===state.planRevision&&world===state.world){if(e.code==='control_conflict')return;if(isExpiredSessionError(e)){markSessionExpired();return;}if(e.httpStatus===503){state.plannerFailureRevision++;setPlannerStatus(false,e.message);updateControlStates();showPlannerUnavailable();if(state.paused)$('flightNote').textContent=`扫描已暂停；路径规划暂不可用：${e.message}`;}else{$('planningBadge').textContent='规划失败';$('statusMessage').textContent=`规划失败：${e.message}`;if(state.paused)$('flightNote').textContent=`扫描已暂停；路径规划失败：${e.message}`;else if(state.flying)$('flightNote').textContent=`路径规划失败：${e.message}；扫描任务仍在继续。`;else $('flightNote').textContent=`路径规划失败：${e.message}`;}} }
    finally {
      if(requestId===state.planRevision&&world===state.world){
        const retryAfterReconnect=state.retryPlanPending&&!responseReceived&&state.connected!==false&&!state.route.length;
        state.retryPlanPending=false;
        if(state.flying)$('flightState').textContent=state.paused?'已暂停':'扫描中';
        state.planning=false; refreshCoverage();
        if(retryAfterReconnect){state.planQueued=false;planRoute();return;}
        if(state.planQueued){state.planQueued=false;planRoute();}
        else if(state.resumePending&&state.flying&&!state.paused){state.resumePending=false;tick();}
      }
    }
  }
  function animateMovement() {
    const movement=state.movement;
    if(!movement||state.paused||state.movementFrame!==null)return;
    state.movementFrame=requestAnimationFrame(timestamp=>{
      state.movementFrame=null;
      if(state.movement!==movement)return;
      if(!state.flying||movement.world!==state.world||movement.revision!==state.worldRevision) { finishMovement(false); return; }
      if(state.paused) { movement.previousFrameTime=null; return; }
      if(movement.previousFrameTime!==null) movement.elapsed+=Math.max(0,timestamp-movement.previousFrameTime);
      movement.previousFrameTime=timestamp;
      const reducedMotion=window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      const progress=reducedMotion?1:Math.min(1,movement.elapsed/movement.duration);
      const previous={...state.dronePosition};
      state.dronePosition={x:movement.from.x+(movement.to.x-movement.from.x)*progress,y:movement.from.y+(movement.to.y-movement.from.y)*progress};
      const travelled=Math.hypot(state.dronePosition.x-previous.x,state.dronePosition.y-previous.y);
      state.flightDistance+=travelled;movement.distanceTravelled+=travelled;
      if(timestamp-(movement.lastDistanceReadoutAt??-Infinity)>=250||progress>=1){
        $('flightDistanceText').textContent=`${(state.flightDistance*(state.world.cell_size_m||100)/1000).toFixed(2)} km`;
        movement.lastDistanceReadoutAt=timestamp;
      }
      if(timestamp-(movement.lastDrawAt??-Infinity)>=1000/30||progress>=1){draw();movement.lastDrawAt=timestamp;}
      if(Math.hypot(state.dronePosition.x-movement.scanCursor.x,state.dronePosition.y-movement.scanCursor.y)>=SCAN_UPDATE_DISTANCE||progress>=1) {
        movement.scanTarget={x:Math.round(state.dronePosition.x),y:Math.round(state.dronePosition.y)};
        movement.scanDistanceTarget=movement.distanceTravelled;
        void flushMovementScan(movement);
      }
      if(progress>=1) {
        movement.arrived=true;movement.scanTarget={x:movement.to.x,y:movement.to.y};
        movement.scanDistanceTarget=movement.distanceTravelled;void flushMovementScan(movement);return;
      }
      animateMovement();
    });
  }
  function finishMovement(arrived) {
    const movement=state.movement;
    if(!movement)return;
    if(state.movementFrame!==null)cancelAnimationFrame(state.movementFrame);
    state.movementFrame=null; state.movement=null;
    if(arrived)state.dronePosition={...movement.to};
    draw(); movement.resolve(arrived);
  }
  function discardUnconfirmedMovementDistance(movement=state.movement) {
    if(!movement)return;
    const distanceTravelled=Number.isFinite(movement.distanceTravelled)?movement.distanceTravelled:null;
    const confirmedDistance=Number.isFinite(movement.confirmedDistance)?movement.confirmedDistance:0;
    const unconfirmed=distanceTravelled===null?
      state.dronePosition&&Math.hypot(
        state.dronePosition.x-(movement.scanCursor?.x??movement.from.x),
        state.dronePosition.y-(movement.scanCursor?.y??movement.from.y),
      ):Math.max(0,distanceTravelled-confirmedDistance);
    if(!Number.isFinite(unconfirmed)||unconfirmed<=0)return;
    state.flightDistance=Math.max(0,state.flightDistance-unconfirmed);
    $('flightDistanceText').textContent=`${(state.flightDistance*(state.world?.cell_size_m||100)/1000).toFixed(2)} km`;
  }
  async function flyTo(target,world,revision,legId) {
    const from=state.dronePosition||world.start, distance=Math.hypot(target.x-from.x,target.y-from.y);
    if(distance===0) {
      try {
        const ok=await sense(from,target,world,revision,legId);
        if(ok){state.dronePosition={...target};refreshCoverage();draw();consumeDeferredPlan(world);}
        return ok;
      } catch(e) {
        if(world===state.world&&revision===state.worldRevision){if(e.code==='scan_request_capacity'){markScanRequestCapacityReached(from,state.flightDistance);return false;}if(e.code==='control_conflict')return false;if(isExpiredSessionError(e)){markSessionExpired();return false;}stopFlight();const unsupported=isUnsupportedScanApiError(e);$('flightState').textContent=unsupported?'服务版本不兼容':'扫描数据待同步';$('flightNote').textContent=unsupported?unsupportedScanApiNote():`传感器读取失败：${e.message}。本段请求已保留；服务恢复后点击“开始自适应扫描”重试并同步数据。`;$('statusMessage').textContent=unsupported?'仿真服务版本过旧；扫描请求已保留':'航段扫描失败，等待结果同步';}
        return false;
      }
    }
    const speed=Math.max(.5,Number($('speedRange').value));
    return new Promise(resolve=>{
      state.movement={from:{...from},to:{...target},legId,scanCursor:{x:Math.round(from.x),y:Math.round(from.y)},scanTarget:{x:Math.round(from.x),y:Math.round(from.y)},scanPending:false,arrived:false,world,revision,elapsed:0,previousFrameTime:null,lastDrawAt:-Infinity,lastDistanceReadoutAt:-Infinity,duration:Math.max(90,distance*(world.cell_size_m||100)/100*45/speed),distanceTravelled:0,confirmedDistance:0,scanDistanceTarget:0,serverConfirmedFlightDistance:state.flightDistance,resolve};
      animateMovement();
    });
  }
  async function tick() {
    if(!state.flying||state.paused||state.worldLoading)return;
    const world=state.world, revision=state.worldRevision;
    if(state.startingFlight){state.resumePending=true;return;}
    if(state.planning||state.tickRunningRevision===revision){state.resumePending=true;return;}
    state.resumePending=false; state.retryScanSelection=false; state.tickRunningRevision=revision; clearTimeout(state.tick); state.tick=null;
    try {
    const endpointRevision=state.endpointRevision;
    const missionSynced=await syncMissionPoints(world,revision);
    if(!missionSynced) {
      if(world===state.world&&revision===state.worldRevision&&endpointRevision!==state.endpointRevision){state.retryScanSelection=true;return;}
      if(world===state.world&&revision===state.worldRevision){stopFlight();$('flightState').textContent='起终点同步失败';$('flightNote').textContent='服务端尚未确认起终点；检查仿真服务后点击“开始自适应扫描”重试。';}
      return;
    }
    if(!state.flying||state.paused||world!==state.world||revision!==state.worldRevision)return;
    let choice;
    try {
      choice=await post('/api/scan/next',{session_id:world.session_id,radius:4,guidance_weight:state.guidanceWeight},15000);
      if(!choice||typeof choice!=='object'||Array.isArray(choice)||!pointInWorld(choice.drone_position,world))throw new Error('扫描服务未返回有效的无人机位置');
      if(choice.target!==null&&(!pointInWorld(choice.target,world)||typeof choice.leg_id!=='string'||!choice.leg_id))throw new Error('扫描服务未返回有效的航段授权');
      if(!Number.isSafeInteger(choice.unknown_cells)||choice.unknown_cells<0||choice.unknown_cells>world.width*world.height)throw new Error('扫描服务未返回有效的未探测格数量');
      state.dronePosition={...choice.drone_position};
    } catch(e) {
      if(world===state.world&&revision===state.worldRevision){if(e.code==='control_conflict')return;if(isExpiredSessionError(e)){markSessionExpired();return;}stopFlight();$('flightState').textContent='扫描中断';$('flightNote').textContent=`选点请求失败，扫描已停止；修复连接或请求错误后，点击“开始自适应扫描”重试。${e.message}`;$('statusMessage').textContent='主动扫描选点失败';}
      return;
    }
    if(!state.flying||state.paused||world!==state.world||revision!==state.worldRevision)return;
    if(endpointRevision!==state.endpointRevision){state.retryScanSelection=true;return;}
    const p=choice.target;
    if(!p) {
      let left=choice.unknown_cells;
      const localUnknown=()=>world.width*world.height-state.observed.reduce((count,row)=>count+row.filter(Boolean).length,0);
      if(left!==localUnknown()) {
        state.scanPath=[];stopFlight();
        if(!await reconcileSessionSnapshot(world,revision)||world!==state.world||revision!==state.worldRevision)return;
        const syncedUnknown=localUnknown();
        if(left!==syncedUnknown) {
          $('flightState').textContent='扫描状态不一致';
          $('flightNote').textContent=`扫描服务报告 ${left} 格未探测；同步后的地图覆盖率为 ${$('coverageText').textContent}（${syncedUnknown} 格未探测）。为避免错误报告完成状态，扫描已停止；可以重试扫描。`;
          $('statusMessage').textContent='扫描状态不一致；地图数据已同步，未宣告完成';
          return;
        }
        left=syncedUnknown;
      }
      if(!left&&state.coverage>=.99&&state.scanSincePlan<5){state.scanSincePlan=5;await planRoute();}
      state.scanPath=[];stopFlight();$('flightState').textContent=left?'扫描候选耗尽':'扫描完成';$('flightNote').textContent=left?`扫描候选点已耗尽，仍有 ${left} 格未探测；可重新启动残余扫描。`:'扫描覆盖已完成；地图不存在未探测格。';$('statusMessage').textContent=left?`扫描候选点已耗尽，仍有 ${left} 格未探测`:'扫描完成，覆盖率达到 100%';return;
    }
    const candidateIndex=state.scanPath.findIndex(q=>q.x===p.x&&q.y===p.y);
    const current=state.dronePosition||world.start;
    $('flightState').textContent='扫描中';
    const endpointsKnownBefore=endpointsKnown();
    if(!await flyTo(p,world,revision,choice.leg_id))return;
    if(!state.flying||state.paused||world!==state.world||revision!==state.worldRevision)return;
    if(candidateIndex>=0)state.scanPath.splice(candidateIndex,1);
    state.previousDronePosition={...current}; state.lastDrone=p; state.scanTrail.push({...p});
    refreshCoverage(); draw();
    const threshold=Math.floor(state.coverage/.20);
    const endpointsJustKnown=endpointRevision===state.endpointRevision&&state.plannedDataRevision!==state.dataRevision&&!endpointsKnownBefore&&endpointsKnown()&&state.coverage>=.20;
    if(threshold>state.scanSincePlan||endpointsJustKnown) {
      state.scanSincePlan=Math.max(threshold,state.scanSincePlan); $('flightState').textContent='融合重规划'; $('flightNote').textContent=`扫描覆盖率达到 ${$('coverageText').textContent}，V5 A* 正在使用新观测更新路径。`;
      void planRoute();
    } else $('flightState').textContent='扫描中';
    publishBlockedEndpointWarning();
    if(state.flying&&!state.paused) { const delay=50/Number($('speedRange').value); state.tick=setTimeout(tick,delay); }
    } finally {
      if(state.tickRunningRevision===revision) {
        state.tickRunningRevision=null;
        if(state.retryScanSelection&&state.flying&&!state.paused) { state.retryScanSelection=false; tick(); }
        else if(state.resumePending&&state.flying&&!state.paused) { state.resumePending=false; tick(); }
      }
    }
  }
  function acquireSessionControl() {
    if(state.controlAcquirePromise)return state.controlAcquirePromise;
    const world=state.world,revision=state.worldRevision;
    if(!world||state.pageSessionReleased||state.sessionSnapshotRequired)return Promise.resolve(false);
    state.controlAcquireError=null;
    state.startingFlight=true;updateControlStates();
    const acquire=(async()=>{
      try {
        if(!state.controlBusy) {
          await state.missionUpdateQueue.catch(()=>false);
          if(world!==state.world||revision!==state.worldRevision)return false;
          if(state.missionSyncedRevision!==state.endpointRevision&&
             !await syncMissionPoints(world,revision))return false;
        }
        const snapshot=await post('/api/session/control',{session_id:world.session_id,action:'acquire',release_token:world.release_token},15000);
        if(world!==state.world||revision!==state.worldRevision||!validSessionSnapshot(snapshot,world))return false;
        if(snapshot.control_token!==state.controlToken||snapshot.control_epoch!==state.controlEpoch||state.controlOwner!==PAGE_ID||state.controlBusy||state.controlLost)return false;
        if(snapshot.control_claimed||snapshot.control_taken_over) {
          if(state.movement) {
            discardUnconfirmedMovementDistance();
            finishMovement(false);
          }
          applySessionSnapshot(snapshot,world);
        } else if(typeof snapshot.scan_request_capacity_reached==='boolean')state.scanRequestCapacityReached=snapshot.scan_request_capacity_reached;
        if(state.scanRequestCapacityReached) {
          markScanRequestCapacityReached(snapshot.drone_position,snapshot.flight_distance);
          return false;
        }
        state.controlBusy=false;state.controlLost=false;
        $('statusMessage').textContent=snapshot.control_taken_over?'已接管地图控制权，并同步最新扫描状态':'地图控制权已确认';
        return true;
      } catch(error) {
        state.controlAcquireError=error;
        if(isExpiredSessionError(error)){markSessionExpired();return false;}
        if(world===state.world&&revision===state.worldRevision&&!error.controlConflict) {
          $('statusMessage').textContent=`无法取得地图控制权：${error.message}`;
        }
        return false;
      } finally {
        if(world===state.world&&revision===state.worldRevision)state.startingFlight=false;
        if(state.controlAcquirePromise===acquire)state.controlAcquirePromise=null;
        updateControlStates();
      }
    })();
    state.controlAcquirePromise=acquire;
    return acquire;
  }
  function finishCompletedScan() {
    stopFlight();
    $('flightState').textContent='扫描完成';
    $('flightNote').textContent='扫描覆盖已完成，地图已全部探测；可查看地图或规划路线。';
    $('statusMessage').textContent='扫描覆盖率已达到 100%，任务完成';
  }
  async function startFlight() {
    if(!state.world||state.worldLoading||state.sessionExpired||state.pageSessionReleased||state.scanRequestCapacityReached||state.sessionSnapshotRequired||state.worldSessionUnconfirmed||(state.coverage>=1&&!state.pendingScan)||state.reconcilingScan||state.startingFlight||state.flying)return;
    const controlAcquired=await acquireSessionControl();
    if(state.scanRequestCapacityReached){markScanRequestCapacityReached(state.dronePosition,state.flightDistance);return;}
    if(!controlAcquired||!state.world||state.worldLoading||state.sessionExpired||state.worldSessionUnconfirmed)return;
    if(state.coverage>=1&&!state.pendingScan){finishCompletedScan();return;}
    if(state.pendingScan) {
      state.reconcilingScan=true; updateControlStates();
      $('flightState').textContent='同步扫描数据'; $('statusMessage').textContent='正在恢复上一段扫描结果…';
      const recovered=await retryPendingScan();
      state.reconcilingScan=false; updateControlStates();
      if(state.scanRequestCapacityReached){markScanRequestCapacityReached(state.dronePosition,state.flightDistance);return;}
      if(!recovered||!state.world||state.worldLoading||state.sessionExpired||state.worldSessionUnconfirmed)return;
      if(state.coverage>=1) {
        $('flightState').textContent='扫描完成'; $('flightNote').textContent='上一段扫描数据已恢复，地图覆盖已完成。';
        $('statusMessage').textContent='扫描结果已同步，覆盖率达到 100%';
        updateControlStates(); return;
      }
      if(state.connected===false)return;
      if(state.coverage>=.20&&endpointsKnown()&&!state.route.length&&!state.planning)void planRoute();
    }
    state.announcedCoverageMilestone=Math.floor(state.coverage*10)*10;
    state.flying=true;state.paused=false;updateControlStates();$('flightState').textContent='扫描中';$('flightNote').textContent='无人机沿自适应路径飞行并持续扫描；覆盖率达到 20% 后自动更新航线。';$('statusMessage').textContent=`无人机开始执行任务走廊引导扫描，当前覆盖率 ${$('coverageText').textContent}`;tick();
  }
  function pauseFlight() {
    if(state.worldLoading||state.startingFlight)return;
    if(!state.paused) {
      state.paused=true;
      state.resumePending=false;
      updateControlStates();
      clearTimeout(state.tick); state.tick=null;
      if(state.movementFrame!==null)cancelAnimationFrame(state.movementFrame);
      state.movementFrame=null;
      if(state.movement)state.movement.previousFrameTime=null;
      const position=state.dronePosition||state.world?.start;
      $('pauseBtn').textContent='▶';$('pauseBtn').setAttribute('aria-label','继续扫描');$('pauseBtn').title='继续扫描';$('flightState').textContent='已暂停';
      $('flightNote').textContent=state.connected===false?'仿真服务离线；扫描已暂停，连接恢复后可以继续。':state.movement?`扫描已暂停；无人机停在 X ${position.x.toFixed(1)}、Y ${position.y.toFixed(1)}；继续后会完成当前航段。`:state.planning?'扫描已暂停；当前规划会在继续扫描后恢复。':'扫描已暂停；继续后将从当前位置选择下一扫描点。';
      $('statusMessage').textContent='扫描已暂停';
      return;
    }
    const world=state.world,revision=state.worldRevision;
    state.paused=false;
    state.resumePending=true;
    updateControlStates();
    $('flightState').textContent=state.planning?'融合重规划':'正在确认';
    $('flightNote').textContent=state.movement?'正在确认地图控制权；确认后继续当前航段。':state.planning?'正在确认地图控制权；确认后继续当前路径规划。':'正在确认地图控制权；确认后恢复扫描。';
    $('statusMessage').textContent=state.planning?'规划继续；正在确认地图控制权':'正在确认地图控制权以继续扫描';
    void (async()=>{
      const controlAcquired=await acquireSessionControl();
      if(state.scanRequestCapacityReached){markScanRequestCapacityReached(state.dronePosition,state.flightDistance);return;}
      if(!controlAcquired||world!==state.world||revision!==state.worldRevision||!state.flying) {
        if(world===state.world&&revision===state.worldRevision&&state.flying&&!state.controlBusy) {
          state.paused=true;state.resumePending=false;updateControlStates();
          $('pauseBtn').textContent='▶';$('pauseBtn').setAttribute('aria-label','继续扫描');$('pauseBtn').title='继续扫描';
          const error=state.controlAcquireError;
          const reason=error?.message||'地图控制权确认失败';
          $('flightState').textContent='已暂停';
          $('flightNote').textContent=error?.transportError||state.connected===false
            ?`控制权确认失败：${reason}；扫描保持暂停，服务恢复后可再次继续。`
            :error?.httpStatus>=500
              ?`控制权确认失败：${reason}；扫描保持暂停，请检查仿真服务后重试。`
              :`控制权确认失败：${reason}；扫描保持暂停，根据拒绝原因修正后可再次继续。`;
        }
        return;
      }
      if(state.coverage>=1&&!state.pendingScan){finishCompletedScan();return;}
      updateControlStates();
      $('pauseBtn').textContent='Ⅱ';$('pauseBtn').setAttribute('aria-label','暂停扫描');$('pauseBtn').title='暂停扫描';$('flightState').textContent=state.planning?'融合重规划':'扫描中';$('statusMessage').textContent='扫描继续';
      $('flightNote').textContent=state.planning?'扫描已继续；系统正在使用最新地形数据更新路径。':'扫描已继续；无人机沿自适应路径飞行并持续扫描，覆盖率达到 20% 后自动更新航线。';
      if(state.movement) animateMovement();
      else if(state.planning)state.resumePending=true;
      else if(state.tickRunningRevision===state.worldRevision)state.resumePending=true;
      else { state.resumePending=false;tick(); }
    })();
  }
  function stopFlight() {
    state.flying=false;state.paused=false;clearTimeout(state.tick);state.tick=null;state.resumePending=false;
    finishMovement(false);
    $('pauseBtn').textContent='Ⅱ';$('pauseBtn').setAttribute('aria-label','暂停扫描');$('pauseBtn').title='暂停扫描';updateControlStates();
  }
  function eventPoint(e) { const r=canvas.getBoundingClientRect(),g=JSON.parse(canvas.dataset.geometry||'{}'),x=e.clientX-r.left-g.ox,y=e.clientY-r.top-g.oy; if(!g.cell||x<0||y<0||x>=g.width*g.cell||y>=g.height*g.cell)return null; return {x:clamp(Math.floor(x/g.cell),0,g.width-1),y:clamp(Math.floor(y/g.cell),0,g.height-1)}; }
  function refreshMapPointerReadout(preferKeyboard=false) {
    if(preferKeyboard&&document.activeElement===canvas&&state.keyboardMapUsed&&!state.pointerReadoutActive&&state.keyboardPoint){$('coordinates').textContent=coordinates(state.keyboardPoint);return;}
    if(!state.world||!state.pointerClient)return;
    const point=eventPoint(state.pointerClient);$('coordinates').textContent=point?coordinates(point):'地图外';
  }
  function endpointsChanged(changedPoint) {
    if(state.worldLoading||state.reconcilingScan||state.controlBusy||state.startingFlight||state.sessionExpired||state.pageSessionReleased||state.sessionSnapshotRequired||state.worldSessionUnconfirmed)return;
    state.endpointRevision++;
    void syncMissionPoints();
    if(changedPoint==='start'&&!state.flying&&!state.paused&&state.coverage===0&&state.flightDistance===0){
      state.dronePosition={...state.world.start};state.previousDronePosition=null;state.lastDrone=null;
      state.scanPath=scanPattern();state.scanTrail=[{...state.world.start}];
    }
    state.planRevision++; state.planning=false; state.planQueued=false; state.retryPlanPending=false;
    state.route=[];
    $('distanceMetric').textContent='—'; $('costMetric').textContent='—'; $('searchMetric').textContent='—'; $('planningBadge').textContent='尚未规划';
    state.scanSincePlan=Math.floor(state.coverage/.20);
    updatePoints(changedPoint); refreshCoverage(); draw();
    state.planAfterScan=!endpointsKnown();
    if(!publishBlockedEndpointWarning()) {
      $('flightNote').textContent=state.connected===false?'仿真服务离线；端点、扫描数据与飞行进度已保留，恢复后将继续规划。':!endpointsKnown()?'端点已更新；继续扫描，扫描到起点和目标点后自动规划。':state.coverage>=.20?'端点已更新；扫描数据与飞行进度已保留，正在重规划路线。':'端点已更新；扫描数据与飞行进度已保留，继续扫描后自动规划。';
      $('statusMessage').textContent=endpointsKnown()?'起终点已设置':'等待扫描到起点和目标点后规划';
    }
    if(state.coverage>=.20&&endpointsKnown()) planRoute();
  }

  function selectPoint(which, focusMap=false) {
    if(state.worldLoading||state.reconcilingScan||state.controlBusy||state.startingFlight||state.sessionExpired||state.pageSessionReleased||state.sessionSnapshotRequired||state.worldSessionUnconfirmed)return;
    state.activePoint=which;
    document.querySelectorAll('[data-point]').forEach(button=>{
      const selected=button.dataset.point===which;
      button.classList.toggle('selected',selected); button.setAttribute('aria-pressed',String(selected));
    });
    $('statusMessage').textContent=which==='start'
      ?focusMap?'已选择起点；方向键移动光标，Enter 设置，也可单击地图':'已选择起点；单击地图设置。键盘操作请先聚焦地图，再用方向键移动光标并按 Enter 设置。'
      :focusMap?'已选择目标点；方向键移动光标，Enter 设置，也可单击地图':'已选择目标点；单击地图设置。键盘操作请先聚焦地图，再用方向键移动光标并按 Enter 设置。';
    if(focusMap)canvas.focus();
  }
  canvas.addEventListener('click',e=>{pointerFocusPending=false;if(!state.world||state.worldLoading||state.reconcilingScan||state.controlBusy||state.startingFlight||state.sessionExpired||state.pageSessionReleased||state.sessionSnapshotRequired||state.worldSessionUnconfirmed)return;const p=eventPoint(e);if(!p)return;if(e.detail!==0){state.pointerReadoutActive=true;state.pointerClient={clientX:e.clientX,clientY:e.clientY};}const which=state.activePoint,previous=state.world[which],changed=previous.x!==p.x||previous.y!==p.y;state.world[which]=p;state.keyboardPoint={...p};$('coordinates').textContent=coordinates(p);announceKeyboardPoint(p,'已设置');if(changed)endpointsChanged(which);});
  canvas.addEventListener('contextmenu',e=>{pointerFocusPending=false;e.preventDefault();if(!state.world||state.worldLoading||state.reconcilingScan||state.controlBusy||state.startingFlight||state.sessionExpired||state.pageSessionReleased||state.sessionSnapshotRequired||state.worldSessionUnconfirmed)return;const p=eventPoint(e);if(!p)return;const previous=state.world.goal,changed=previous.x!==p.x||previous.y!==p.y;state.world.goal=p;state.keyboardPoint={...p};$('coordinates').textContent=coordinates(p);selectPoint('goal');announceKeyboardPoint(p,'已设置');if(changed)endpointsChanged('goal');});
  canvas.addEventListener('mousemove',e=>{if(state.world){state.pointerReadoutActive=true;state.pointerClient={clientX:e.clientX,clientY:e.clientY};const p=eventPoint(state.pointerClient);$('coordinates').textContent=p?coordinates(p):'地图外';}});
  canvas.addEventListener('mouseleave',()=>{state.pointerClient=null;state.pointerReadoutActive=false;const keyboardFocus=document.activeElement===canvas&&state.keyboardMapUsed&&state.keyboardPoint;$('coordinates').textContent=keyboardFocus?coordinates(state.keyboardPoint):'地图外';});
  const mapViewport=$('mapViewport');
  mapViewport.addEventListener('scroll',()=>refreshMapPointerReadout(true));
  mapViewport.addEventListener('keydown',e=>{
    if(e.target!==mapViewport)return;
    const delta=40,steps={ArrowLeft:[-delta,0],ArrowRight:[delta,0],ArrowUp:[0,-delta],ArrowDown:[0,delta]},step=steps[e.key];
    if(!step)return;
    const maxLeft=Math.max(0,mapViewport.scrollWidth-mapViewport.clientWidth),maxTop=Math.max(0,mapViewport.scrollHeight-mapViewport.clientHeight);
    if((step[0]&&!maxLeft)||(step[1]&&!maxTop))return;
    e.preventDefault();mapViewport.scrollLeft=clamp(mapViewport.scrollLeft+step[0],0,maxLeft);mapViewport.scrollTop=clamp(mapViewport.scrollTop+step[1],0,maxTop);
  });
  let pointerFocusPending=false;
  canvas.addEventListener('pointerdown',()=>{pointerFocusPending=true;});
  canvas.addEventListener('pointercancel',()=>{pointerFocusPending=false;});
  canvas.addEventListener('focus',()=>{
    const pointerFocus=pointerFocusPending;pointerFocusPending=false;state.pointerReadoutActive=false;
    if(!pointerFocus){state.keyboardPoint={...(state.world?.[state.activePoint]||{x:0,y:0})};state.keyboardMapUsed=true;revealKeyboardPoint(state.keyboardPoint);$('coordinates').textContent=coordinates(state.keyboardPoint);announceKeyboardPoint(state.keyboardPoint);}
    draw();
  });
  canvas.addEventListener('blur',()=>{pointerFocusPending=false;draw();});
  canvas.addEventListener('keydown',e=>{
    if(!state.world||state.worldLoading||state.reconcilingScan||state.controlBusy||state.startingFlight||state.sessionExpired||state.pageSessionReleased||state.sessionSnapshotRequired||state.worldSessionUnconfirmed)return;
    const p=state.keyboardPoint||{...state.world[state.activePoint]},next={...p};
    if(e.key==='ArrowLeft')next.x=Math.max(0,p.x-1);else if(e.key==='ArrowRight')next.x=Math.min(state.world.width-1,p.x+1);else if(e.key==='ArrowUp')next.y=Math.max(0,p.y-1);else if(e.key==='ArrowDown')next.y=Math.min(state.world.height-1,p.y+1);
    else if(e.key==='Enter'||e.key===' '){e.preventDefault();state.pointerReadoutActive=false;const which=state.activePoint,previous=state.world[which],changed=previous.x!==p.x||previous.y!==p.y;state.world[which]={...p};announceKeyboardPoint(p,'已设置');if(changed)endpointsChanged(which);return;}else return;
    state.keyboardMapUsed=true;e.preventDefault();state.pointerReadoutActive=false;state.keyboardPoint=next;revealKeyboardPoint(next);$('coordinates').textContent=coordinates(next);announceKeyboardPoint(next);draw();
  });
  document.querySelectorAll('[data-point]').forEach(b=>b.addEventListener('click',e=>selectPoint(b.dataset.point,e.detail===0)));
  $('layerSelect').addEventListener('change',()=>{draw();if(state.keyboardMapUsed&&state.keyboardPoint)announceKeyboardPoint(state.keyboardPoint,'当前格');}); $('randomBtn').addEventListener('click',()=>{const pending=state.pendingWorldCreate;void makeWorld(pending?.seed??Math.floor(Math.random()*900000)+100000,pending);}); $('planBtn').addEventListener('click',planRoute); $('startBtn').addEventListener('click',startFlight); $('pauseBtn').addEventListener('click',pauseFlight); $('resetBtn').addEventListener('click',()=>{const pending=state.pendingWorldCreate;void makeWorld(pending?.seed??state.worldRequestSeed,pending);}); $('zoomOutBtn').addEventListener('click',()=>setMapZoom(state.mapZoom-.5)); $('zoomInBtn').addEventListener('click',()=>setMapZoom(state.mapZoom+.5)); $('zoomFitBtn').addEventListener('click',resetMapZoom); $('speedRange').addEventListener('input',e=>{$('speedText').textContent=`${Number(e.target.value).toFixed(1)}×`;}); window.addEventListener('resize',resizeMap); window.addEventListener('scroll',()=>refreshMapPointerReadout(true),{passive:true});
  window.addEventListener('pagehide',event=>{
    if(event.persisted)return;
    const world=state.world;
    if(typeof world?.session_id!=='string'||!world.session_id||typeof world.release_token!=='string'||!world.release_token)return;
    try { window.navigator?.sendBeacon?.('/api/session/release',new Blob([JSON.stringify({session_id:world.session_id,release_token:world.release_token,page_id:PAGE_ID,control_token:state.controlToken})],{type:'application/json'})); } catch {}
  });
  window.addEventListener('pageshow',event=>{if(event.persisted)void checkServer();});
  for(const which of ['start','goal']) {
    const suffix=which==='start'?'Start':'Goal';
    $(`apply${suffix}Coordinates`).addEventListener('click',()=>applyCoordinateInputs(which));
    for(const axis of ['X','Y']) $(`${which}${axis}`).addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();applyCoordinateInputs(which);}});
  }
  selectPoint(state.activePoint); setInterval(checkServer,10000); void initializeSimulator();
})();
