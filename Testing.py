
# """
# Self-contained HTML for the Linear Regression interactive walkthrough.
# Covers: What is Linear Regression, Cost Function (MSE), Gradient Descent,
# Model Evaluation (R², Residuals), and Multiple Linear Regression.
# Embed in Streamlit via st.components.v1.html(LR_VISUAL_HTML, height=LR_VISUAL_HEIGHT).
# """
#
# LR_VISUAL_HTML = """
# <!DOCTYPE html>
# <html>
# <head>
# <meta charset="utf-8"/>
# <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
# <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
# <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
# <style>
#   * { margin: 0; padding: 0; box-sizing: border-box; }
#   body { background: #0a0a0f; overflow-x: hidden; }
#   input[type="range"] { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: #1e1e2e; outline: none; }
#   input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; background: #4ecdc4; }
#   @keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
#   @keyframes fadein { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
# </style>
# </head>
# <body>
# <div id="root"></div>
# <script type="text/babel">
#
# var useState = React.useState;
# var useEffect = React.useEffect;
# var useMemo = React.useMemo;
# var useRef = React.useRef;
#
# var C = {
#   bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
#   accent: "#4ecdc4", blue: "#38bdf8", purple: "#a78bfa",
#   yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
#   dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
#   cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
#   teal: "#4ecdc4",
# };
#
# var ARR = "\\u2192";
# var DASH = "\\u2014";
# var BULB = "\\uD83D\\uDCA1";
# var TARG = "\\uD83C\\uDFAF";
# var WARN = "\\u26A0";
# var LARR = "\\u2190";
# var DELTA = "\\u0394";
# var THETA = "\\u03B8";
# var ALPHA = "\\u03B1";
# var SIGMA = "\\u03A3";
# var HAT = "\\u0302";
# var SUP2 = "\\u00B2";
# var PARTIAL = "\\u2202";
# var NABLA = "\\u2207";
# var APPROX = "\\u2248";
# var SQRT = "\\u221A";
# var PM = "\\u00B1";
# var INF = "\\u221E";
# var CHECK = "\\u2713";
# var CROSS = "\\u2717";
#
# function TabBar(props) {
#   var tabs = props.tabs, active = props.active, onChange = props.onChange;
#   return (
#     <div style={{ display: "flex", gap: 0, borderBottom: "2px solid " + C.border, marginBottom: 24, overflowX: "auto" }}>
#       {tabs.map(function(t, i) {
#         return (
#           <button key={i} onClick={function() { onChange(i); }} style={{
#             padding: "12px 18px", background: "none", border: "none",
#             borderBottom: active === i ? "2px solid " + C.accent : "2px solid transparent",
#             color: active === i ? C.accent : C.muted, cursor: "pointer",
#             fontSize: 11, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
#             transition: "all 0.2s", whiteSpace: "nowrap", marginBottom: -2,
#           }}>
#             {t}
#           </button>
#         );
#       })}
#     </div>
#   );
# }
#
# function Card(props) {
#   return (
#     <div style={Object.assign({
#       background: C.card, borderRadius: 10, padding: "18px 22px",
#       border: "1px solid " + (props.highlight ? C.accent : C.border),
#       transition: "border 0.3s",
#     }, props.style || {})}>
#       {props.children}
#     </div>
#   );
# }
#
# function Insight(props) {
#   return (
#     <div style={Object.assign({
#       maxWidth: 750, margin: "16px auto 0",
#       padding: "16px 22px", background: "rgba(78,205,196,0.06)",
#       borderRadius: 10, border: "1px solid rgba(78,205,196,0.2)",
#     }, props.style || {})}>
#       <div style={{ fontSize: 11, fontWeight: 700, color: C.accent, marginBottom: 6 }}>{(props.icon || BULB) + " " + (props.title || "Key Insight")}</div>
#       <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.8 }}>{props.children}</div>
#     </div>
#   );
# }
#
# function SectionTitle(props) {
#   return (
#     <div style={{ textAlign: "center", marginBottom: 20 }}>
#       <div style={{ fontSize: 18, fontWeight: 800, color: C.text, marginBottom: 4 }}>{props.title}</div>
#       <div style={{ fontSize: 12, color: C.muted }}>{props.subtitle}</div>
#     </div>
#   );
# }
#
# function Slider(props) {
#   return (
#     <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:8 }}>
#       <div style={{ fontSize:10, color:C.muted, width:60, textAlign:"right", fontFamily:"monospace" }}>{props.label}</div>
#       <input type="range" min={props.min} max={props.max} step={props.step || 0.01}
#         value={props.value} onChange={function(e){ props.onChange(parseFloat(e.target.value)); }}
#         style={{ flex:1 }} />
#       <div style={{ fontSize:10, color:C.accent, width:48, fontFamily:"monospace", fontWeight:700 }}>
#         {typeof props.value === "number" ? props.value.toFixed(props.decimals !== undefined ? props.decimals : 2) : props.value}
#       </div>
#     </div>
#   );
# }
#
# /* ===============================================================
#    DATA POINTS (shared across tabs)
#    =============================================================== */
# var BASE_POINTS = [
#   {x:1.2, y:2.1}, {x:2.0, y:3.8}, {x:2.8, y:4.2}, {x:3.5, y:5.9},
#   {x:4.1, y:6.1}, {x:4.9, y:7.5}, {x:5.6, y:8.0}, {x:6.2, y:9.3},
#   {x:7.0, y:10.1},{x:7.8, y:11.4},{x:8.5, y:12.0},{x:9.1, y:13.2},
# ];
#
# // SVG viewport helpers
# var VW = 440, VH = 300;
# var PAD = { l:44, r:16, t:16, b:38 };
# var PW = VW - PAD.l - PAD.r;
# var PH = VH - PAD.t - PAD.b;
#
# function scaleX(x) { return PAD.l + ((x - 0) / 11) * PW; }
# function scaleY(y) { return PAD.t + PH - ((y - 0) / 16) * PH; }
#
# function PlotAxes(props) {
#   var title = props.title || "";
#   var xlabel = props.xlabel || "x";
#   var ylabel = props.ylabel || "y";
#   var xTicks = [0,2,4,6,8,10];
#   var yTicks = [0,4,8,12,16];
#   return (
#     <g>
#       {/* grid */}
#       {xTicks.map(function(v){ return <line key={"gx"+v} x1={scaleX(v)} y1={PAD.t} x2={scaleX(v)} y2={PAD.t+PH} stroke={C.border} strokeWidth={0.5} />; })}
#       {yTicks.map(function(v){ return <line key={"gy"+v} x1={PAD.l} y1={scaleY(v)} x2={PAD.l+PW} y2={scaleY(v)} stroke={C.border} strokeWidth={0.5} />; })}
#       {/* axes */}
#       <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={PAD.t+PH} stroke={C.dim} strokeWidth={1.5} />
#       <line x1={PAD.l} y1={PAD.t+PH} x2={PAD.l+PW} y2={PAD.t+PH} stroke={C.dim} strokeWidth={1.5} />
#       {/* ticks & labels */}
#       {xTicks.map(function(v){ return <text key={"tx"+v} x={scaleX(v)} y={PAD.t+PH+13} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{v}</text>; })}
#       {yTicks.map(function(v){ return <text key={"ty"+v} x={PAD.l-6} y={scaleY(v)+3} textAnchor="end" fill={C.muted} fontSize={8} fontFamily="monospace">{v}</text>; })}
#       <text x={PAD.l+PW/2} y={VH-4} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{xlabel}</text>
#       <text x={10} y={PAD.t+PH/2} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace" transform={"rotate(-90,10,"+(PAD.t+PH/2)+")"}>{ylabel}</text>
#       {title && <text x={PAD.l+PW/2} y={PAD.t-4} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">{title}</text>}
#     </g>
#   );
# }
#
# /* ===============================================================
#    TAB 1: WHAT IS LINEAR REGRESSION
#    =============================================================== */
# function TabIntro() {
#   var _s = useState(0.8); var slope = _s[0]; var setSlope = _s[1];
#   var _i = useState(1.0); var intercept = _i[0]; var setIntercept = _i[1];
#
#   var trueSlope = 1.42;
#   var trueIntercept = 0.6;
#
#   function predict(x) { return slope * x + intercept; }
#   function truePredict(x) { return trueSlope * x + trueIntercept; }
#
#   var mse = useMemo(function() {
#     var sum = 0;
#     BASE_POINTS.forEach(function(p) { var e = p.y - predict(p.x); sum += e*e; });
#     return sum / BASE_POINTS.length;
#   }, [slope, intercept]);
#
#   var bestMse = useMemo(function() {
#     var sum = 0;
#     BASE_POINTS.forEach(function(p) { var e = p.y - truePredict(p.x); sum += e*e; });
#     return sum / BASE_POINTS.length;
#   }, []);
#
#   var xStart = 0, xEnd = 10;
#   var y0 = predict(xStart), y1 = predict(xEnd);
#   var ty0 = truePredict(xStart), ty1 = truePredict(xEnd);
#
#   return (
#     <div>
#       <SectionTitle title="What is Linear Regression?" subtitle={"Find the line that best explains the relationship between x and y"} />
#
#       <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center",maxWidth:750,margin:"0 auto 16px"}}>
#         <Card style={{flex:"1 1 420px"}}>
#           <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Interactive Fit</div>
#           <svg width="100%" viewBox={"0 0 "+VW+" "+VH} style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border,display:"block"}}>
#             <PlotAxes xlabel="Feature (x)" ylabel="Target (y)" />
#             {/* residual lines */}
#             {BASE_POINTS.map(function(p,i){
#               var yhat = predict(p.x);
#               return <line key={i} x1={scaleX(p.x)} y1={scaleY(p.y)} x2={scaleX(p.x)} y2={scaleY(yhat)}
#                 stroke={C.red} strokeWidth={1} strokeDasharray="3,2" opacity={0.5} />;
#             })}
#             {/* true best-fit line (dashed) */}
#             <line x1={scaleX(xStart)} y1={scaleY(ty0)} x2={scaleX(xEnd)} y2={scaleY(ty1)}
#               stroke={C.green} strokeWidth={1.5} strokeDasharray="6,4" opacity={0.6} />
#             <text x={scaleX(9.5)} y={scaleY(ty1)-8} fill={C.green} fontSize={8} textAnchor="middle" fontFamily="monospace" opacity={0.8}>best fit</text>
#             {/* user line */}
#             <line x1={scaleX(xStart)} y1={scaleY(y0)} x2={scaleX(xEnd)} y2={scaleY(y1)}
#               stroke={C.accent} strokeWidth={2.5} />
#             {/* data points */}
#             {BASE_POINTS.map(function(p,i){
#               return <circle key={i} cx={scaleX(p.x)} cy={scaleY(p.y)} r={4}
#                 fill={C.blue} stroke={C.bg} strokeWidth={1.5} />;
#             })}
#             {/* equation label */}
#             <rect x={PAD.l+4} y={PAD.t+4} width={140} height={20} rx={4} fill={C.bg} opacity={0.85} />
#             <text x={PAD.l+10} y={PAD.t+17} fill={C.accent} fontSize={10} fontFamily="monospace" fontWeight={700}>
#               {"y\u0302 = " + slope.toFixed(2) + "x + " + intercept.toFixed(2)}
#             </text>
#           </svg>
#           <div style={{marginTop:12}}>
#             <Slider label="slope m" min={-0.5} max={3} step={0.05} value={slope} onChange={setSlope} />
#             <Slider label="intercept b" min={-4} max={6} step={0.1} value={intercept} onChange={setIntercept} />
#           </div>
#         </Card>
#
#         <div style={{flex:"1 1 240px",display:"flex",flexDirection:"column",gap:12}}>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:6}}>THE EQUATION</div>
#             <div style={{fontSize:22,fontWeight:800,color:C.accent,fontFamily:"monospace",textAlign:"center",margin:"8px 0"}}>
#               y = mx + b
#             </div>
#             <div style={{fontSize:9,color:C.muted,lineHeight:1.9}}>
#               <div><span style={{color:C.accent,fontWeight:700}}>y</span> {DASH} predicted output</div>
#               <div><span style={{color:C.blue,fontWeight:700}}>x</span> {DASH} input feature</div>
#               <div><span style={{color:C.yellow,fontWeight:700}}>m</span> {DASH} slope (weight)</div>
#               <div><span style={{color:C.purple,fontWeight:700}}>b</span> {DASH} intercept (bias)</div>
#             </div>
#           </Card>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:6}}>YOUR LINE vs BEST FIT</div>
#             <div style={{textAlign:"center",margin:"8px 0"}}>
#               <div style={{fontSize:11,color:C.muted,marginBottom:4}}>Your MSE</div>
#               <div style={{fontSize:28,fontWeight:800,color:mse < bestMse*1.15 ? C.green : mse < bestMse*2 ? C.yellow : C.red}}>
#                 {mse.toFixed(2)}
#               </div>
#               <div style={{fontSize:9,color:C.dim,marginTop:4}}>Best possible: {bestMse.toFixed(2)}</div>
#             </div>
#             <div style={{background:"#08080d",borderRadius:6,padding:"6px 10px",marginTop:4}}>
#               <div style={{fontSize:9,color:C.muted,marginBottom:4}}>HOW CLOSE?</div>
#               <div style={{height:8,borderRadius:4,background:C.border,overflow:"hidden"}}>
#                 <div style={{
#                   height:"100%",borderRadius:4,
#                   background:mse<bestMse*1.15?C.green:mse<bestMse*2?C.yellow:C.red,
#                   width:Math.max(4,Math.min(100,(1-(mse-bestMse)/(20))*100))+"%",
#                   transition:"all 0.3s"
#                 }} />
#               </div>
#             </div>
#           </Card>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:6}}>RESIDUALS</div>
#             <div style={{fontSize:10,color:C.muted,lineHeight:1.8}}>
#               The <span style={{color:C.red}}>red dashed lines</span> are <span style={{color:C.text,fontWeight:700}}>residuals</span>: the vertical gap between each data point and your predicted line.
#             </div>
#             <div style={{marginTop:8,fontSize:10,color:C.muted,lineHeight:1.8}}>
#               Residual<sub>i</sub> = y<sub>i</sub> <span style={{color:C.red}}>{DASH}</span> {"y\u0302"}<sub>i</sub>
#             </div>
#           </Card>
#         </div>
#       </div>
#
#       <Insight icon={BULB} title="The Core Idea">
#         Linear regression finds the values of <span style={{color:C.yellow,fontWeight:700}}>m</span> (slope) and{" "}
#         <span style={{color:C.purple,fontWeight:700}}>b</span> (intercept) that minimize the total squared residuals.
#         Drag the sliders above to feel how the line changes {DASH} the <span style={{color:C.green}}>green dashed line</span> shows the mathematically optimal fit.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 2: COST FUNCTION
#    =============================================================== */
# function TabCost() {
#   var _m = useState(1.42); var slope = _m[0]; var setSlope = _m[1];
#
#   var trueSlope = 1.42;
#   var trueIntercept = 0.6;
#
#   function mseAt(m) {
#     var sum = 0;
#     BASE_POINTS.forEach(function(p) {
#       var yhat = m * p.x + trueIntercept;
#       var e = p.y - yhat;
#       sum += e * e;
#     });
#     return sum / BASE_POINTS.length;
#   }
#
#   var currentMse = mseAt(slope);
#
#   // Build MSE curve over slope range
#   var curvePoints = [];
#   for (var m = -0.5; m <= 3.5; m += 0.05) {
#     curvePoints.push({ m: m, mse: mseAt(m) });
#   }
#
#   // Cost curve SVG
#   var CW = 440, CH = 260;
#   var CP = { l:50, r:20, t:24, b:40 };
#   var CPW = CW - CP.l - CP.r;
#   var CPH = CH - CP.t - CP.b;
#
#   var mMin = -0.5, mMax = 3.5;
#   var costMax = 80;
#
#   function cx(m) { return CP.l + ((m - mMin) / (mMax - mMin)) * CPW; }
#   function cy(cost) { return CP.t + CPH - Math.min(1, cost / costMax) * CPH; }
#
#   var pathD = curvePoints.map(function(pt, i) {
#     return (i === 0 ? "M" : "L") + cx(pt.m).toFixed(1) + "," + cy(pt.mse).toFixed(1);
#   }).join(" ");
#
#   var mTicks = [-0.5, 0.5, 1.42, 2.5, 3.5];
#   var costTicks = [0, 20, 40, 60, 80];
#
#   return (
#     <div>
#       <SectionTitle title="The Cost Function (MSE)" subtitle={"We need a way to measure how wrong our line is " + DASH + " then minimize it"} />
#
#       <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center",maxWidth:750,margin:"0 auto 16px"}}>
#         <Card style={{flex:"1 1 420px"}}>
#           <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>MSE vs. Slope</div>
#           <svg width="100%" viewBox={"0 0 "+CW+" "+CH} style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border,display:"block"}}>
#             {/* grid */}
#             {mTicks.map(function(v){ return <line key={"gm"+v} x1={cx(v)} y1={CP.t} x2={cx(v)} y2={CP.t+CPH} stroke={C.border} strokeWidth={0.5} />; })}
#             {costTicks.map(function(v){ return <line key={"gc"+v} x1={CP.l} y1={cy(v)} x2={CP.l+CPW} y2={cy(v)} stroke={C.border} strokeWidth={0.5} />; })}
#             {/* axes */}
#             <line x1={CP.l} y1={CP.t} x2={CP.l} y2={CP.t+CPH} stroke={C.dim} strokeWidth={1.5} />
#             <line x1={CP.l} y1={CP.t+CPH} x2={CP.l+CPW} y2={CP.t+CPH} stroke={C.dim} strokeWidth={1.5} />
#             {mTicks.map(function(v){ return <text key={"tm"+v} x={cx(v)} y={CP.t+CPH+13} textAnchor="middle" fill={v===1.42?C.green:C.muted} fontSize={8} fontFamily="monospace" fontWeight={v===1.42?700:400}>{v===1.42?"m*":v.toFixed(1)}</text>; })}
#             {costTicks.map(function(v){ return <text key={"tc"+v} x={CP.l-6} y={cy(v)+3} textAnchor="end" fill={C.muted} fontSize={8} fontFamily="monospace">{v}</text>; })}
#             <text x={CP.l+CPW/2} y={CH-4} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">slope (m)</text>
#             <text x={12} y={CP.t+CPH/2} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace" transform={"rotate(-90,12,"+(CP.t+CPH/2)+")"}>MSE (cost)</text>
#             {/* parabola */}
#             <path d={pathD} fill="none" stroke={C.purple} strokeWidth={2.5} />
#             {/* minimum marker */}
#             <circle cx={cx(trueSlope)} cy={cy(mseAt(trueSlope))} r={6} fill={C.green} opacity={0.9} />
#             <text x={cx(trueSlope)+10} y={cy(mseAt(trueSlope))-8} fill={C.green} fontSize={9} fontFamily="monospace" fontWeight={700}>global minimum</text>
#             {/* current slope marker */}
#             <line x1={cx(slope)} y1={CP.t} x2={cx(slope)} y2={CP.t+CPH} stroke={C.accent} strokeWidth={1} strokeDasharray="4,3" opacity={0.7} />
#             <circle cx={cx(slope)} cy={cy(currentMse)} r={6} fill={C.accent} stroke={C.bg} strokeWidth={1.5} />
#             <text x={CP.l+8} y={CP.t+14} fill={C.muted} fontSize={8} fontFamily="monospace">MSE = {currentMse.toFixed(2)}</text>
#           </svg>
#           <div style={{marginTop:12}}>
#             <Slider label="slope m" min={-0.5} max={3.5} step={0.05} value={slope} onChange={setSlope} />
#           </div>
#         </Card>
#
#         <div style={{flex:"1 1 240px",display:"flex",flexDirection:"column",gap:12}}>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:8}}>MEAN SQUARED ERROR</div>
#             <div style={{background:"#08080d",borderRadius:8,padding:"12px",textAlign:"center",fontFamily:"monospace",marginBottom:10}}>
#               <div style={{fontSize:11,color:C.purple,fontWeight:700,lineHeight:2}}>
#                 MSE = (1/n) {SIGMA}(y{"\u1D62"} {DASH} {"y\u0302"}{"\u1D62"}){SUP2}
#               </div>
#             </div>
#             <div style={{fontSize:9,color:C.muted,lineHeight:1.9}}>
#               <div><span style={{color:C.text,fontWeight:700}}>n</span> {DASH} number of data points</div>
#               <div><span style={{color:C.blue,fontWeight:700}}>y{"\u1D62"}</span> {DASH} actual value</div>
#               <div><span style={{color:C.accent,fontWeight:700}}>{"y\u0302"}{"\u1D62"}</span> {DASH} predicted value</div>
#               <div><span style={{color:C.yellow,fontWeight:700}}>squaring</span> {DASH} penalizes large errors</div>
#             </div>
#           </Card>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:6}}>CURRENT MSE</div>
#             <div style={{textAlign:"center"}}>
#               <div style={{fontSize:36,fontWeight:800,color:currentMse < 0.5 ? C.green : currentMse < 5 ? C.yellow : C.red, transition:"color 0.3s"}}>
#                 {currentMse.toFixed(2)}
#               </div>
#               <div style={{fontSize:9,color:C.dim,marginTop:4}}>
#                 {currentMse < 0.5 ? CHECK+" near optimal!" : currentMse < 5 ? "getting closer..." : WARN+" far from optimal"}
#               </div>
#             </div>
#             <div style={{height:8,borderRadius:4,background:C.border,overflow:"hidden",marginTop:8}}>
#               <div style={{
#                 height:"100%",borderRadius:4,
#                 background:currentMse<0.5?C.green:currentMse<5?C.yellow:C.red,
#                 width:Math.max(4,100-Math.min(100,(currentMse/80)*100))+"%",
#                 transition:"all 0.3s"
#               }} />
#             </div>
#           </Card>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:6}}>WHY SQUARED?</div>
#             <div style={{fontSize:10,color:C.muted,lineHeight:1.8}}>
#               Squaring errors means <span style={{color:C.red}}>large mistakes</span> are penalized{" "}
#               <span style={{color:C.text,fontWeight:700}}>much more heavily</span> than small ones.
#               It also makes the cost function <span style={{color:C.purple}}>smooth and differentiable</span>, enabling calculus-based optimization.
#             </div>
#           </Card>
#         </div>
#       </div>
#
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Individual Squared Errors</div>
#         <svg width="100%" viewBox={"0 0 680 80"} style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border,display:"block"}}>
#           {BASE_POINTS.map(function(p,i){
#             var yhat = slope * p.x + trueIntercept;
#             var err = p.y - yhat;
#             var sq = err * err;
#             var maxSq = 50;
#             var barH = Math.min(60, (sq / maxSq) * 60);
#             var bx = 20 + i * 54;
#             var barColor = sq < 0.5 ? C.green : sq < 5 ? C.yellow : C.red;
#             return (
#               <g key={i}>
#                 <rect x={bx} y={70-barH} width={40} height={barH} rx={3}
#                   fill={barColor} opacity={0.7} />
#                 <text x={bx+20} y={76} textAnchor="middle" fill={C.muted} fontSize={7} fontFamily="monospace">x={p.x}</text>
#                 {sq > 1 && <text x={bx+20} y={70-barH-4} textAnchor="middle" fill={barColor} fontSize={7} fontFamily="monospace">{sq.toFixed(1)}</text>}
#               </g>
#             );
#           })}
#         </svg>
#       </Card>
#
#       <Insight icon={TARG} title="The Parabola">
#         The cost function forms a <span style={{color:C.purple,fontWeight:700}}>U-shaped parabola</span> (convex). This is crucial {DASH} it means there is exactly{" "}
#         <span style={{color:C.green,fontWeight:700}}>one global minimum</span> and no local minima to get stuck in.
#         Drag the slider to the bottom of the bowl to find <span style={{color:C.accent}}>m*</span>!
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 3: GRADIENT DESCENT
#    =============================================================== */
# function TabGradient() {
#   var _lr = useState(0.3); var lr = _lr[0]; var setLr = _lr[1];
#   var _step = useState(0); var step = _step[0]; var setStep = _step[1];
#   var _hist = useState([{m:3.2, mse:0}]); var history = _hist[0]; var setHistory = _hist[1];
#   var _running = useState(false); var running = _running[0]; var setRunning = _running[1];
#   var intervalRef = useRef(null);
#
#   var trueIntercept = 0.6;
#
#   function mseAt(m) {
#     var sum = 0;
#     BASE_POINTS.forEach(function(p) {
#       var e = p.y - (m * p.x + trueIntercept);
#       sum += e * e;
#     });
#     return sum / BASE_POINTS.length;
#   }
#
#   function gradientAt(m) {
#     var sum = 0;
#     BASE_POINTS.forEach(function(p) {
#       var e = p.y - (m * p.x + trueIntercept);
#       sum += -2 * p.x * e;
#     });
#     return sum / BASE_POINTS.length;
#   }
#
#   function doStep(hist) {
#     var last = hist[hist.length - 1];
#     var grad = gradientAt(last.m);
#     var newM = last.m - lr * grad;
#     var newMse = mseAt(newM);
#     return hist.concat([{m: newM, mse: newMse, grad: grad}]);
#   }
#
#   useEffect(function() {
#     if (running) {
#       intervalRef.current = setInterval(function() {
#         setHistory(function(h) {
#           if (h.length >= 40) { setRunning(false); return h; }
#           return doStep(h);
#         });
#         setStep(function(s) { return s + 1; });
#       }, 300);
#     } else {
#       clearInterval(intervalRef.current);
#     }
#     return function() { clearInterval(intervalRef.current); };
#   }, [running, lr]);
#
#   function reset() {
#     setRunning(false);
#     clearInterval(intervalRef.current);
#     setHistory([{m: 3.2, mse: mseAt(3.2)}]);
#     setStep(0);
#   }
#
#   function stepOnce() {
#     setHistory(function(h) { return doStep(h); });
#     setStep(function(s) { return s + 1; });
#   }
#
#   var current = history[history.length - 1];
#
#   // Cost curve
#   var CW2 = 440, CH2 = 260;
#   var CP2 = { l:50, r:20, t:24, b:40 };
#   var CPW2 = CW2 - CP2.l - CP2.r;
#   var CPH2 = CH2 - CP2.t - CP2.b;
#   var mMin = -0.5, mMax = 3.8;
#   var costMax = 80;
#
#   function cx2(m) { return CP2.l + ((m - mMin) / (mMax - mMin)) * CPW2; }
#   function cy2(cost) { return CP2.t + CPH2 - Math.min(1, cost / costMax) * CPH2; }
#
#   var curvePoints2 = [];
#   for (var m = mMin; m <= mMax; m += 0.05) {
#     curvePoints2.push({ m: m, mse: mseAt(m) });
#   }
#   var pathD2 = curvePoints2.map(function(pt, i) {
#     return (i === 0 ? "M" : "L") + cx2(pt.m).toFixed(1) + "," + cy2(pt.mse).toFixed(1);
#   }).join(" ");
#
#   var converged = history.length > 5 && Math.abs(current.mse - mseAt(1.42)) < 0.05;
#
#   return (
#     <div>
#       <SectionTitle title="Gradient Descent" subtitle={"Iteratively nudge m in the direction that reduces MSE the fastest"} />
#
#       <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center",maxWidth:750,margin:"0 auto 16px"}}>
#         <Card style={{flex:"1 1 420px"}}>
#           <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Descending the Cost Curve</div>
#           <svg width="100%" viewBox={"0 0 "+CW2+" "+CH2} style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border,display:"block"}}>
#             {/* grid */}
#             {[-0.5,0.5,1.42,2.5,3.5].map(function(v){ return <line key={"g2m"+v} x1={cx2(v)} y1={CP2.t} x2={cx2(v)} y2={CP2.t+CPH2} stroke={C.border} strokeWidth={0.5} />; })}
#             {[0,20,40,60,80].map(function(v){ return <line key={"g2c"+v} x1={CP2.l} y1={cy2(v)} x2={CP2.l+CPW2} y2={cy2(v)} stroke={C.border} strokeWidth={0.5} />; })}
#             <line x1={CP2.l} y1={CP2.t} x2={CP2.l} y2={CP2.t+CPH2} stroke={C.dim} strokeWidth={1.5} />
#             <line x1={CP2.l} y1={CP2.t+CPH2} x2={CP2.l+CPW2} y2={CP2.t+CPH2} stroke={C.dim} strokeWidth={1.5} />
#             {[-0.5,0.5,1.42,2.5,3.5].map(function(v){ return <text key={"t2m"+v} x={cx2(v)} y={CP2.t+CPH2+13} textAnchor="middle" fill={v===1.42?C.green:C.muted} fontSize={8} fontFamily="monospace" fontWeight={v===1.42?700:400}>{v===1.42?"m*":v.toFixed(1)}</text>; })}
#             {[0,20,40,60,80].map(function(v){ return <text key={"t2c"+v} x={CP2.l-6} y={cy2(v)+3} textAnchor="end" fill={C.muted} fontSize={8} fontFamily="monospace">{v}</text>; })}
#             <text x={CP2.l+CPW2/2} y={CH2-4} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">slope (m)</text>
#             <text x={12} y={CP2.t+CPH2/2} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace" transform={"rotate(-90,12,"+(CP2.t+CPH2/2)+")"}>MSE</text>
#             {/* parabola */}
#             <path d={pathD2} fill="none" stroke={C.purple} strokeWidth={2.5} opacity={0.7} />
#             {/* minimum */}
#             <circle cx={cx2(1.42)} cy={cy2(mseAt(1.42))} r={5} fill={C.green} opacity={0.9} />
#             {/* path of descent */}
#             {history.map(function(h,i){
#               if (i===0) return null;
#               var prev = history[i-1];
#               return <line key={i}
#                 x1={cx2(prev.m)} y1={cy2(prev.mse)}
#                 x2={cx2(h.m)} y2={cy2(h.mse)}
#                 stroke={C.accent} strokeWidth={1.5} opacity={0.6} />;
#             })}
#             {/* current position */}
#             <circle cx={cx2(current.m)} cy={cy2(current.mse)} r={7}
#               fill={converged ? C.green : C.accent} stroke={C.bg} strokeWidth={2} />
#             {/* gradient arrow */}
#             {current.grad && !converged && (function(){
#               var gradDir = current.grad > 0 ? -1 : 1;
#               var arrowLen = 30;
#               return <line
#                 x1={cx2(current.m)} y1={cy2(current.mse)}
#                 x2={cx2(current.m)+gradDir*arrowLen} y2={cy2(current.mse)}
#                 stroke={C.orange} strokeWidth={2}
#                 markerEnd="url(#arrowhead)" />;
#             })()}
#             <defs>
#               <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="3" refY="2" orient="auto">
#                 <polygon points="0 0, 6 2, 0 4" fill={C.orange} />
#               </marker>
#             </defs>
#             <text x={CP2.l+8} y={CP2.t+14} fill={converged?C.green:C.muted} fontSize={8} fontFamily="monospace" fontWeight={700}>
#               {converged ? CHECK+" converged! m=" + current.m.toFixed(3) : "step "+history.length+" | m="+current.m.toFixed(3)}
#             </text>
#           </svg>
#
#           <div style={{display:"flex",gap:8,marginTop:12,flexWrap:"wrap"}}>
#             <button onClick={function(){setRunning(function(r){return !r;});}} style={{
#               padding:"8px 16px",borderRadius:8,border:"1px solid "+C.accent,
#               background:running?"rgba(78,205,196,0.15)":C.accent+"25",
#               color:C.accent,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"
#             }}>{running ? "⏸ Pause" : "▶ Run"}</button>
#             <button onClick={stepOnce} style={{
#               padding:"8px 16px",borderRadius:8,border:"1px solid "+C.blue,
#               background:C.blue+"15",color:C.blue,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"
#             }}>+1 Step</button>
#             <button onClick={reset} style={{
#               padding:"8px 16px",borderRadius:8,border:"1px solid "+C.dim,
#               background:"transparent",color:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"
#             }}>↺ Reset</button>
#           </div>
#           <div style={{marginTop:10}}>
#             <Slider label={ALPHA+" lr"} min={0.01} max={0.8} step={0.01} value={lr} onChange={function(v){reset();setLr(v);}} />
#           </div>
#         </Card>
#
#         <div style={{flex:"1 1 240px",display:"flex",flexDirection:"column",gap:12}}>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:8}}>THE UPDATE RULE</div>
#             <div style={{background:"#08080d",borderRadius:8,padding:"12px",fontFamily:"monospace",textAlign:"center"}}>
#               <div style={{fontSize:11,color:C.accent,fontWeight:700,lineHeight:2.2}}>
#                 m := m {DASH} {ALPHA} {PARTIAL}J/{PARTIAL}m
#               </div>
#               <div style={{fontSize:8,color:C.muted,marginTop:4}}>
#                 where {PARTIAL}J/{PARTIAL}m = {DASH}(2/n){SIGMA}x{"\u1D62"}(y{"\u1D62"}{DASH}{"y\u0302"}{"\u1D62"})
#               </div>
#             </div>
#           </Card>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:8}}>CURRENT STATE</div>
#             {[
#               {l:"slope m", v:current.m.toFixed(4), c:converged?C.green:C.accent},
#               {l:"MSE", v:current.mse.toFixed(4), c:converged?C.green:C.yellow},
#               {l:"gradient", v:current.grad?current.grad.toFixed(4):"—", c:C.orange},
#               {l:"steps", v:history.length, c:C.blue},
#             ].map(function(row,i){
#               return (
#                 <div key={i} style={{display:"flex",justifyContent:"space-between",padding:"4px 0",borderBottom:"1px solid "+C.border,fontSize:10,fontFamily:"monospace"}}>
#                   <span style={{color:C.muted}}>{row.l}</span>
#                   <span style={{color:row.c,fontWeight:700}}>{row.v}</span>
#                 </div>
#               );
#             })}
#           </Card>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:8}}>LEARNING RATE EFFECTS</div>
#             {[
#               {lr:"too small", desc:"Slow convergence, many steps needed", c:C.blue},
#               {lr:"just right", desc:"Smooth descent to minimum", c:C.green},
#               {lr:"too large", desc:"Overshoots, may diverge!", c:C.red},
#             ].map(function(row,i){
#               return (
#                 <div key={i} style={{display:"flex",gap:8,alignItems:"flex-start",marginBottom:6}}>
#                   <div style={{fontSize:8,color:row.c,fontWeight:700,fontFamily:"monospace",width:60,flexShrink:0}}>{row.lr}</div>
#                   <div style={{fontSize:9,color:C.muted}}>{row.desc}</div>
#                 </div>
#               );
#             })}
#           </Card>
#         </div>
#       </div>
#
#       <Insight icon={NABLA} title="The Gradient">
#         The gradient tells us the <span style={{color:C.orange,fontWeight:700}}>slope of the cost curve</span> at our current position. We always move in the{" "}
#         <span style={{color:C.accent,fontWeight:700}}>opposite direction</span> of the gradient {DASH} downhill toward the minimum.
#         Try a very large learning rate ({APPROX}0.8) to see it overshoot!
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 4: MODEL EVALUATION
#    =============================================================== */
# function TabEval() {
#   var _noise = useState(1.2); var noise = _noise[0]; var setNoise = _noise[1];
#
#   var trueSlope = 1.42, trueIntercept = 0.6;
#
#   var points = useMemo(function() {
#     var rng = function(seed) {
#       var x = Math.sin(seed) * 10000;
#       return x - Math.floor(x);
#     };
#     return BASE_POINTS.map(function(p, i) {
#       var n = (rng(i * 7.3 + 1.1) - 0.5) * 2 * noise;
#       return { x: p.x, y: trueSlope * p.x + trueIntercept + n };
#     });
#   }, [noise]);
#
#   var yMean = useMemo(function() {
#     return points.reduce(function(s,p){return s+p.y;},0)/points.length;
#   }, [points]);
#
#   var ssTot = useMemo(function() {
#     return points.reduce(function(s,p){return s+Math.pow(p.y-yMean,2);},0);
#   }, [points, yMean]);
#
#   var ssRes = useMemo(function() {
#     return points.reduce(function(s,p){
#       var yhat = trueSlope * p.x + trueIntercept;
#       return s + Math.pow(p.y - yhat, 2);
#     },0);
#   }, [points]);
#
#   var r2 = Math.max(0, 1 - ssRes / ssTot);
#   var mse = ssRes / points.length;
#   var rmse = Math.sqrt(mse);
#
#   var xStart = 0, xEnd = 10;
#   var y0 = trueIntercept, y1 = trueSlope * xEnd + trueIntercept;
#
#   return (
#     <div>
#       <SectionTitle title="Model Evaluation" subtitle={"Once we have a line, how do we know if it's actually good?"} />
#
#       <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center",maxWidth:750,margin:"0 auto 16px"}}>
#         <Card style={{flex:"1 1 420px"}}>
#           <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Fit & Residuals</div>
#           <svg width="100%" viewBox={"0 0 "+VW+" "+VH} style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border,display:"block"}}>
#             <PlotAxes xlabel="Feature (x)" ylabel="Target (y)" />
#             {/* mean line */}
#             <line x1={scaleX(0)} y1={scaleY(yMean)} x2={scaleX(10)} y2={scaleY(yMean)}
#               stroke={C.muted} strokeWidth={1} strokeDasharray="4,3" opacity={0.5} />
#             <text x={scaleX(9.5)} y={scaleY(yMean)-6} fill={C.muted} fontSize={8} fontFamily="monospace">ȳ</text>
#             {/* residual lines */}
#             {points.map(function(p,i){
#               var yhat = trueSlope * p.x + trueIntercept;
#               return <line key={i} x1={scaleX(p.x)} y1={scaleY(p.y)} x2={scaleX(p.x)} y2={scaleY(yhat)}
#                 stroke={C.red} strokeWidth={1} strokeDasharray="3,2" opacity={0.6} />;
#             })}
#             {/* regression line */}
#             <line x1={scaleX(xStart)} y1={scaleY(y0)} x2={scaleX(xEnd)} y2={scaleY(y1)}
#               stroke={C.accent} strokeWidth={2.5} />
#             {/* data points */}
#             {points.map(function(p,i){
#               return <circle key={i} cx={scaleX(p.x)} cy={scaleY(p.y)} r={4}
#                 fill={C.blue} stroke={C.bg} strokeWidth={1.5} />;
#             })}
#           </svg>
#           <div style={{marginTop:12}}>
#             <Slider label="noise" min={0.1} max={4} step={0.1} value={noise} onChange={setNoise} />
#           </div>
#         </Card>
#
#         <div style={{flex:"1 1 240px",display:"flex",flexDirection:"column",gap:12}}>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:10}}>KEY METRICS</div>
#             {[
#               {label:"R² Score", value:r2.toFixed(3), desc:"% variance explained", color:r2>0.9?C.green:r2>0.7?C.yellow:C.red},
#               {label:"MSE", value:mse.toFixed(3), desc:"Mean Squared Error", color:C.purple},
#               {label:"RMSE", value:rmse.toFixed(3), desc:"Root MSE (same units)", color:C.blue},
#             ].map(function(m,i){
#               return (
#                 <div key={i} style={{padding:"8px 0",borderBottom:"1px solid "+C.border,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
#                   <div>
#                     <div style={{fontSize:10,fontWeight:700,color:C.text,fontFamily:"monospace"}}>{m.label}</div>
#                     <div style={{fontSize:8,color:C.muted}}>{m.desc}</div>
#                   </div>
#                   <div style={{fontSize:20,fontWeight:800,color:m.color,fontFamily:"monospace"}}>{m.value}</div>
#                 </div>
#               );
#             })}
#           </Card>
#
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:8}}>R² DECOMPOSITION</div>
#             <div style={{background:"#08080d",borderRadius:8,padding:"10px",fontFamily:"monospace",fontSize:9,textAlign:"center"}}>
#               <div style={{color:C.accent,fontWeight:700,lineHeight:2}}>R² = 1 {DASH} SS_res / SS_tot</div>
#               <div style={{color:C.muted,lineHeight:1.9}}>
#                 <span style={{color:C.red}}>SS_res</span> = {SIGMA}(y{DASH}{"y\u0302"}){SUP2} = {ssRes.toFixed(1)}<br/>
#                 <span style={{color:C.blue}}>SS_tot</span> = {SIGMA}(y{DASH}ȳ){SUP2} = {ssTot.toFixed(1)}
#               </div>
#             </div>
#             <div style={{marginTop:8}}>
#               <div style={{fontSize:9,color:C.muted,marginBottom:4}}>R² = {r2.toFixed(3)}</div>
#               <div style={{height:10,borderRadius:5,background:C.border,overflow:"hidden"}}>
#                 <div style={{height:"100%",width:(r2*100)+"%",borderRadius:5,background:r2>0.9?C.green:r2>0.7?C.yellow:C.red,transition:"all 0.3s"}} />
#               </div>
#             </div>
#           </Card>
#
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:6}}>INTERPRETING R²</div>
#             {[
#               {range:"0.9 {DASH} 1.0",qual:"Excellent",c:C.green},
#               {range:"0.7 {DASH} 0.9",qual:"Good",c:C.yellow},
#               {range:"0.5 {DASH} 0.7",qual:"Moderate",c:C.orange},
#               {range:"< 0.5",qual:"Poor",c:C.red},
#             ].map(function(row,i){
#               return (
#                 <div key={i} style={{display:"flex",justifyContent:"space-between",padding:"3px 0",fontSize:9,fontFamily:"monospace"}}>
#                   <span style={{color:C.muted}}>{row.range.replace("{DASH}",DASH)}</span>
#                   <span style={{color:row.c,fontWeight:700}}>{row.qual}</span>
#                 </div>
#               );
#             })}
#           </Card>
#         </div>
#       </div>
#
#       <Insight icon={TARG} title="The Noise Slider">
#         Increase the <span style={{color:C.yellow,fontWeight:700}}>noise</span> to simulate messier real-world data.
#         Notice how <span style={{color:C.accent}}>R²</span> drops as variance increases {DASH} the line still fits as well as possible,
#         but the underlying noise limits how much of the variance any linear model can explain.
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    TAB 5: MULTIPLE LINEAR REGRESSION
#    =============================================================== */
# function TabMultiple() {
#   var _sel = useState(0); var sel = _sel[0]; var setSel = _sel[1];
#
#   var scenarios = [
#     {
#       name:"House Prices",
#       color:C.accent,
#       target:"Price ($k)",
#       features:["Size (sqft)","Bedrooms","Distance to city","Age (yrs)","Has garage"],
#       weights:[0.18, 12.0, -2.5, -0.8, 15.0],
#       intercept: 80,
#       desc:"Predict house sale price from structural and location features.",
#       insight:"Garage presence has a big positive weight; distance to city hurts value most per unit.",
#     },
#     {
#       name:"Salary Prediction",
#       color:C.purple,
#       target:"Salary ($k)",
#       features:["Years exp.","Education lvl","Skills score","Industry code","Company size"],
#       weights:[4.2, 8.5, 1.1, 3.0, 0.9],
#       intercept: 35,
#       desc:"Predict annual salary from professional background features.",
#       insight:"Education level and years of experience are the strongest salary predictors here.",
#     },
#     {
#       name:"Student Score",
#       color:C.yellow,
#       target:"Exam Score",
#       features:["Study hrs/day","Sleep hrs","Attendance%","Practice tests","Stress level"],
#       weights:[8.0, 3.5, 0.4, 5.5, -6.0],
#       intercept: 20,
#       desc:"Predict exam scores from student habits and wellbeing indicators.",
#       insight:"Stress level has a large negative weight. Sleep and study hours both matter significantly.",
#     },
#   ];
#
#   var sc = scenarios[sel];
#   var maxAbs = Math.max.apply(null, sc.weights.map(Math.abs));
#
#   var exSample = [2.1, 0.5, -0.8, 1.2, 0.3];
#   var prediction = sc.intercept + sc.weights.reduce(function(s,w,i){ return s + w * exSample[i]; }, 0);
#
#   return (
#     <div>
#       <SectionTitle title="Multiple Linear Regression" subtitle={"Extend to n features: y\u0302 = w\u2081x\u2081 + w\u2082x\u2082 + ... + w\u2099x\u2099 + b"} />
#
#       <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:20,flexWrap:"wrap"}}>
#         {scenarios.map(function(s,i){
#           var on = sel===i;
#           return (
#             <button key={i} onClick={function(){setSel(i);}} style={{
#               padding:"8px 18px",borderRadius:8,border:"1.5px solid "+(on?s.color:C.border),
#               background:on?s.color+"20":C.card,color:on?s.color:C.muted,
#               cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"
#             }}>{s.name}</button>
#           );
#         })}
#       </div>
#
#       <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center",maxWidth:750,margin:"0 auto 16px"}}>
#         <Card style={{flex:"1 1 380px"}} highlight>
#           <div style={{fontSize:11,fontWeight:700,color:sc.color,marginBottom:4}}>{sc.name}</div>
#           <div style={{fontSize:9,color:C.muted,marginBottom:14}}>{sc.desc}</div>
#
#           <div style={{fontSize:9,color:C.muted,marginBottom:8}}>FEATURE WEIGHTS (COEFFICIENTS)</div>
#           {sc.features.map(function(feat, i) {
#             var w = sc.weights[i];
#             var pct = (Math.abs(w) / maxAbs) * 100;
#             var isPos = w >= 0;
#             return (
#               <div key={i} style={{marginBottom:8}}>
#                 <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
#                   <span style={{fontSize:9,color:C.text,fontFamily:"monospace"}}>{feat}</span>
#                   <span style={{fontSize:9,color:isPos?C.green:C.red,fontWeight:700,fontFamily:"monospace"}}>
#                     {isPos?"+":""}{w.toFixed(1)}
#                   </span>
#                 </div>
#                 <div style={{height:7,borderRadius:4,background:C.border,overflow:"hidden"}}>
#                   <div style={{
#                     height:"100%",width:pct+"%",borderRadius:4,
#                     background:isPos?sc.color:C.red,
#                     opacity:0.75,transition:"width 0.4s"
#                   }} />
#                 </div>
#               </div>
#             );
#           })}
#
#           <div style={{marginTop:12,padding:"8px 12px",background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
#             <div style={{fontSize:8,color:C.muted,marginBottom:4}}>INTERCEPT (bias)</div>
#             <div style={{fontSize:14,fontWeight:700,color:sc.color,fontFamily:"monospace"}}>b = {sc.intercept}</div>
#           </div>
#         </Card>
#
#         <div style={{flex:"1 1 260px",display:"flex",flexDirection:"column",gap:12}}>
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:8}}>THE EQUATION</div>
#             <div style={{background:"#08080d",borderRadius:8,padding:"10px",textAlign:"center",fontFamily:"monospace"}}>
#               <div style={{fontSize:10,color:sc.color,fontWeight:700,lineHeight:2.2}}>
#                 y = w{"\u1D40"}x + b
#               </div>
#               <div style={{fontSize:8,color:C.muted,lineHeight:1.9}}>
#                 {"= \u03A3 w\u1D62 \u00B7 x\u1D62 + b"}
#               </div>
#             </div>
#             <div style={{fontSize:9,color:C.muted,lineHeight:1.8,marginTop:8}}>
#               In vector form: multiply the weight vector <span style={{color:sc.color}}>w</span> by the feature vector{" "}
#               <span style={{color:C.blue}}>x</span> and add bias <span style={{color:C.purple}}>b</span>.
#             </div>
#           </Card>
#
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:8}}>EXAMPLE PREDICTION</div>
#             <div style={{fontSize:9,color:C.muted,marginBottom:6}}>Sample feature values:</div>
#             {sc.features.map(function(f,i){
#               return (
#                 <div key={i} style={{display:"flex",justifyContent:"space-between",fontSize:9,fontFamily:"monospace",padding:"2px 0"}}>
#                   <span style={{color:C.muted}}>{f}</span>
#                   <span style={{color:C.blue}}>{exSample[i]}</span>
#                 </div>
#               );
#             })}
#             <div style={{marginTop:10,padding:"8px 12px",background:sc.color+"10",borderRadius:8,border:"1px solid "+sc.color+"30",textAlign:"center"}}>
#               <div style={{fontSize:8,color:C.muted,marginBottom:2}}>PREDICTED {sc.target.toUpperCase()}</div>
#               <div style={{fontSize:22,fontWeight:800,color:sc.color,fontFamily:"monospace"}}>
#                 {prediction.toFixed(1)}
#               </div>
#             </div>
#           </Card>
#
#           <Card>
#             <div style={{fontSize:9,color:C.muted,marginBottom:8}}>ASSUMPTIONS</div>
#             {[
#               {label:"Linearity", c:C.green},
#               {label:"Independence of errors", c:C.green},
#               {label:"Homoscedasticity", c:C.yellow},
#               {label:"Normality of errors", c:C.blue},
#               {label:"No multicollinearity", c:C.orange},
#             ].map(function(a,i){
#               return (
#                 <div key={i} style={{display:"flex",alignItems:"center",gap:6,padding:"3px 0",fontSize:9,fontFamily:"monospace"}}>
#                   <div style={{width:8,height:8,borderRadius:2,background:a.c,flexShrink:0}} />
#                   <span style={{color:C.muted}}>{a.label}</span>
#                 </div>
#               );
#             })}
#           </Card>
#         </div>
#       </div>
#
#       <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
#         <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>From 1D to nD</div>
#         <div style={{display:"flex",gap:0,alignItems:"center",flexWrap:"wrap",justifyContent:"center"}}>
#           {[
#             {label:"Simple",sub:"y = mx + b",dim:"1 feature",c:C.accent},
#             {label:ARR,sub:"",dim:"",c:C.muted},
#             {label:"Multiple",sub:"y = w\u2081x\u2081+w\u2082x\u2082+b",dim:"2 features",c:C.purple},
#             {label:ARR,sub:"",dim:"",c:C.muted},
#             {label:"General",sub:"y = w\u1D40x + b",dim:"n features",c:C.yellow},
#           ].map(function(item,i){
#             if(item.label===ARR) return <div key={i} style={{fontSize:20,color:C.dim,padding:"0 8px"}}>{ARR}</div>;
#             return (
#               <div key={i} style={{textAlign:"center",padding:"10px 16px",background:"#08080d",borderRadius:8,border:"1px solid "+item.c+"30",margin:"4px"}}>
#                 <div style={{fontSize:11,fontWeight:800,color:item.c}}>{item.label}</div>
#                 <div style={{fontSize:9,color:C.muted,fontFamily:"monospace",margin:"4px 0"}}>{item.sub}</div>
#                 <div style={{fontSize:8,color:C.dim}}>{item.dim}</div>
#               </div>
#             );
#           })}
#         </div>
#       </Card>
#
#       <Insight icon={BULB} title="Weights = Importance">
#         A <span style={{color:C.green,fontWeight:700}}>large positive weight</span> means the feature strongly pushes the prediction up.
#         A <span style={{color:C.red,fontWeight:700}}>large negative weight</span> strongly pushes it down.
#         But be careful {DASH} weights are only directly comparable when features are on the{" "}
#         <span style={{color:sc.color,fontWeight:700}}>same scale</span> (normalized). Always standardize before comparing weights!
#       </Insight>
#     </div>
#   );
# }
#
#
# /* ===============================================================
#    ROOT APP
#    =============================================================== */
# function App() {
#   var _t = useState(0); var tab = _t[0]; var setTab = _t[1];
#   var tabs = ["Intro & Fitting", "Cost Function", "Gradient Descent", "Evaluation (R²)", "Multiple Regression"];
#   return (
#     <div style={{ background:C.bg, minHeight:"100vh", padding:"24px 16px", fontFamily:"'JetBrains Mono','SF Mono',monospace", color:C.text, maxWidth:960, margin:"0 auto" }}>
#       <div style={{textAlign:"center",marginBottom:16}}>
#         <div style={{ fontSize:22, fontWeight:800, background:"linear-gradient(135deg,"+C.accent+","+C.blue+")", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", display:"inline-block" }}>Linear Regression</div>
#         <div style={{fontSize:11,color:C.muted,marginTop:4}}>{"Interactive visual walkthrough " + DASH + " from fitting a line to gradient descent to R\u00B2"}</div>
#       </div>
#       <TabBar tabs={tabs} active={tab} onChange={setTab} />
#       {tab===0 && <TabIntro />}
#       {tab===1 && <TabCost />}
#       {tab===2 && <TabGradient />}
#       {tab===3 && <TabEval />}
#       {tab===4 && <TabMultiple />}
#     </div>
#   );
# }
#
# ReactDOM.createRoot(document.getElementById("root")).render(<App />);
#
# </script>
# </body>
# </html>
# """
#
# LR_VISUAL_HEIGHT = 1100


# """Module 00 · Supervised Learning Core Idea"""
#
# DISPLAY_NAME = "00 · Supervised Learning Core Idea"
# ICON = "📈"
# SUBTITLE = "The Fundemental of Supervised Learning"
#
# THEORY = """
#
# ## Supervised Learning in Machine Learning
#
# **Formal Definition:**
# Supervised learning is the task of learning a function f(x; θ) that maps inputs x to outputs y by minimizing
# a loss function over a set of labeled training examples. The parameters θ (weights) are adjusted until the
# model's predictions are as close as possible to the known correct answers.
#
# The fundamental idea behind supervised learning is learning from labeled examples.
# You train a model on a dataset where the correct answers are already known, so the model can learn the mapping between
# inputs and outputs — then apply that learned mapping to new, unseen data.
#
# Think of it like a student learning with an answer key.
# The student sees many practice problems along with their correct answers, learns the patterns,
# and can then answer new problems on their own.
#
# ---
#
# #### How It Works
#
# The process generally follows these steps:
#
#     1. Collect labeled data — Each example in your dataset has an input (called features) and a known output (called a label or target).
#        For example, thousands of emails already tagged as "spam" or "not spam."
#
#     2. Train the model — The algorithm repeatedly makes predictions on the training data and compares them to the known
#        correct answers. The difference between the prediction and the true answer is called the loss or error.
#
#     3. Minimize the error — The model adjusts its internal parameters (weights) to reduce that error over time,
#        using techniques like gradient descent.
#
#     4. Generalize — Once trained, the model applies what it learned to new, unlabeled data it has never seen before.
#
# ---
#
# #### Two Main Types of Supervised Learning
#
#     * Classification — The output is a category. Examples include spam detection (spam/not spam),
#       image recognition (cat/dog/bird), and disease diagnosis (positive/negative).
#       Common algorithms include logistic regression, decision trees, support vector machines, and neural networks.
#
#     * Regression — The output is a continuous numerical value. Examples include predicting house prices,
#       forecasting stock values, or estimating a patient's age from an X-ray.
#       Common algorithms include linear regression and gradient boosted trees.
#
# ---
#
# #### Why "Supervised"?
#
# The term comes from the idea that a human supervisor provides the correct answers during training.
# This is in contrast to unsupervised learning (where there are no labels and the model finds hidden structure on its own)
# and reinforcement learning (where the model learns through trial, error, and reward signals).
#
# ---
#
# #### Three Things to Keep Separate
#
# As you learn supervised learning, it helps to keep three distinct concepts from blurring together:
#
# **The learning paradigm** — Supervised learning is defined by the presence of labeled data and the goal of learning
# an input-to-output mapping. This is the category of problem, not the method of solving it.
#
# **The optimization method** — How the model actually adjusts its parameters. Gradient descent is the most common
# method, but it is not the only one and not part of the definition of supervised learning. Linear regression can be
# solved analytically using the normal equation (no iterations needed). Decision trees use greedy split optimization.
# SVMs use convex optimization. KNN requires no optimization at all — it simply memorizes the data.
# Gradient descent is one tool among several.
#
# **The model class** — The type of function being learned: linear models, trees, ensembles, neural networks, and so on.
# Each makes different assumptions about the structure of the problem.
#
# Keeping these three layers separate prevents a very common confusion: thinking that supervised learning *means*
# gradient descent, or that neural networks are *the* definition of the field.
#
# ---
#
# #### Training Process:
#
# **My Query to Claude:**
#
# so supervised learning algorithms are basically we have to gather high quality data and train the model which detects
# the connections between data points and detect patterns and behaviours and then we once the model is trained we tweak
# the model with updating internal parameters  -  i am assuming this is something like back propagation and then save the
# model for testing data and validation data and also future data
#
#
# **Training Loop**
# On the training loop, you're essentially right. The model makes a prediction,
# compares it to the known label via a loss function, and then backpropagation computes how much each internal parameter (weight)
# contributed to that error. The optimizer (like gradient descent or Adam) then nudges those weights in the direction that
# reduces the error. This cycle repeats thousands or millions of times across your training data.
#
# On backpropagation specifically, your intuition is correct — it's the core mechanism for updating weights in neural networks.
# It works by flowing the error signal backwards through the network layer by layer using the chain rule from calculus,
# figuring out each weight's "blame" for the error. However, it's worth noting that backprop is specific to neural networks.
# Other supervised algorithms like decision trees or linear regression update their parameters differently — through
# mathematical optimization rather than gradient-based backprop.
#
# On the data splits, you're also right that you hold out separate portions of data.
# The typical split is training data (what the model learns from), validation data (used during training to tune
# hyperparameters and catch overfitting early), and test data (a final blind evaluation that simulates real-world unseen data).
# The model never "learns" from the validation or test sets — they purely measure performance.
#
#
# ##### What Are the Internal Weights?
#
# Weights are simply numbers stored inside the model that determine how it transforms inputs into outputs.
# They have no meaning at the start — they're usually initialized randomly — and the entire training process is about
# finding the right values for them.
#
# In a neural network specifically, weights exist in a few forms:
#
# **Connection weights** — Every connection between two neurons has a weight.
# It's a multiplier that says "how much should this signal matter?" A high positive weight amplifies a signal,
# a near-zero weight ignores it, and a negative weight suppresses it.
#
# **Biases** — Each neuron also has a bias term, which is an offset that shifts the output up or down independent of the input.
# It gives the model more flexibility, similar to the intercept in a linear equation like y = mx + b.
#
# In other model types, the "weights" have different names but the same concept applies.
# In linear regression they're called coefficients. In decision trees they're the threshold values at each split.
# In SVMs they define the decision boundary. The universal idea is: these are the numbers the algorithm tunes to fit the data.
#
# At scale, a large neural network can have billions of these weights. GPT-style models are essentially just enormous
# collections of numbers that were tuned on massive datasets until the patterns in language emerged.
#
# ---
#
# Key Challenges
# The main practical challenges in supervised learning are getting enough high-quality labeled data (labeling is expensive and time-consuming),
# avoiding overfitting (where the model memorizes the training data but fails on new data),
# and choosing the right model architecture for the problem.
#
# One thing worth adding to your mental model is the concept of overfitting vs. generalization.
# A model can get very good at detecting patterns in training data but fail on new data because it essentially memorized
# the training examples rather than learning the underlying pattern.
#
# This is why the validation set is so important — it acts as an early warning system for that problem.
#
# The pipeline is:
#
# **quality data → training loop (forward pass → loss → backprop → weight update) → validate and tune → test on held-out data → deploy on future real-world data.**
#
# ---
#
# #### The Bias–Variance Tradeoff
#
# Overfitting and underfitting are not two separate problems — they are two sides of a single fundamental tension
# called the bias-variance tradeoff. Understanding this framing gives you a much sharper mental model for diagnosing
# model behaviour.
#
# **Bias** is the error introduced by the model's assumptions about the data. A high-bias model is too simple —
# it cannot capture the real structure in the data and performs poorly even on the training set.
# This is called underfitting. A linear model trying to fit a clearly curved relationship is a classic example.
#
# **Variance** is the error introduced by the model being too sensitive to the specific training data it was shown.
# A high-variance model fits the training data extremely well but fails on new data because it has learned the noise
# and quirks of those specific examples rather than the underlying pattern. This is overfitting.
#
# **The tradeoff** is that reducing one tends to increase the other. As you make a model more complex
# (more layers, more trees, more features), bias decreases but variance increases.
# As you simplify the model, variance decreases but bias increases.
# The goal is to find the sweet spot where both are low enough that the model generalises well.
#
#     High Bias (Underfitting)    →   Model too simple, misses real patterns
#                                     Training error: HIGH
#                                     Test error:     HIGH
#
#     High Variance (Overfitting) →   Model too complex, memorises noise
#                                     Training error: LOW
#                                     Test error:     HIGH
#
#     Optimal Model               →   Captures real structure, ignores noise
#                                     Training error: LOW (acceptable)
#                                     Test error:     LOW (close to training)
#
# Regularization techniques (L1, L2, Dropout) and ensemble methods (Random Forest, Gradient Boosting) are
# largely tools for managing this tradeoff — they constrain model complexity to reduce variance without
# sacrificing too much bias.
#
# ---
#
# #### Loss vs. Metric — Two Different Things
#
# This distinction is small but important enough to state explicitly because the two are often confused.
#
# **Loss** is what the model actually optimises during training. It is chosen for its mathematical properties —
# it must be differentiable so gradients can be computed. Examples include Mean Squared Error for regression
# and Cross-Entropy for classification. The raw loss value is often not human-interpretable on its own.
#
# **Metric** is how you evaluate model performance for the real-world use case. It is chosen to reflect
# what actually matters for the problem. Examples include Accuracy, F1 Score, AUC-ROC for classification,
# and RMSE or MAE for regression.
#
# A practical example: you train a fraud detection model using binary cross-entropy loss (because it's
# differentiable and works well for probabilities), but you evaluate it using F1 Score and AUC-ROC
# (because raw accuracy is misleading when only 0.1% of transactions are fraudulent).
#
#     Train with:   Cross-Entropy Loss   →  the optimisation target
#     Report on:    F1 Score / AUC       →  the business performance target
#
# The loss going down during training is a good sign, but it is the metric — evaluated on the held-out
# test set — that tells you whether the model is actually useful.
#
# ---
#
# #### The IID Assumption — Why Your Data Distribution Matters
#
# Supervised learning rests on a quiet but critical assumption that is almost never stated explicitly in
# introductory courses: your training data and your real-world deployment data must come from the same
# underlying distribution.
#
# This is called the IID assumption — Independent and Identically Distributed.
#
# **Independent** means each training example was drawn separately and does not influence the others.
# **Identically Distributed** means all examples — including future unseen ones — are drawn from the
# same underlying distribution.
#
# When this assumption holds, a model that generalises well on the test set will also generalise well
# in production. When it breaks, everything can fall apart.
#
# **Distribution shift** is what happens when the real-world data your deployed model sees is
# meaningfully different from what it was trained on. This is one of the most common causes of
# model degradation in production. Examples include:
#
#     * A credit scoring model trained on pre-2020 data deployed during a recession —
#       spending patterns have shifted.
#
#     * A medical image classifier trained on data from one hospital deployed at another —
#       scanner hardware and imaging protocols differ.
#
#     * A sentiment analysis model trained on product reviews applied to tweets —
#       language style and vocabulary differ significantly.
#
# In all of these cases the model's training and test performance may look excellent, but real-world
# performance collapses because the distribution has shifted. This is why monitoring model performance
# in production — not just at evaluation time — is a core part of the ML engineering discipline.
#
# The practical implication: always ask "does my training data reflect the conditions my model will
# actually face in deployment?" before trusting your test metrics.
#
# ---
#
# ## **The Broader Landscape of Supervised Models**
#
# **Linear Models:**
# These are the simplest and oldest. Linear Regression assumes the relationship between input and output is a straight line —
# it just finds the best-fit line through the data by optimizing coefficients. Logistic Regression despite the name is a
# classification algorithm — it uses a sigmoid function to squash the output into a probability between 0 and 1.
# These models are highly interpretable, fast to train, and still widely used in finance, medicine, and anywhere explainability matters.
#
# **Ensemble Methods**
# These are where traditional ML gets really powerful — the idea is combining many weak models into one strong model.
#
# **Random Forest**
# Random Forest builds hundreds of decision trees, each trained on a random subset of the data and features,
# then averages their predictions. The randomness prevents any one tree from overfitting and the ensemble is much more robust.
#
# **Gradient Boosting** - (XGBoost, LightGBM, CatBoost) takes a different approach — it builds trees sequentially,
# where each new tree specifically focuses on correcting the errors the previous trees made.
# This is an extremely powerful technique and XGBoost in particular dominated machine learning competitions for years.
# For tabular/structured data (spreadsheets, databases), gradient boosting often beats neural networks.
#
# **Support Vector Machines (SVMs)**
# SVMs find the optimal decision boundary between classes by maximizing the margin — the gap between the boundary and the
# nearest data points from each class. They're particularly clever because through something called the kernel trick,
# they can handle non-linear boundaries without explicitly transforming the data. Very effective on smaller datasets and
# high-dimensional data like text.
#
# **K-Nearest Neighbors (KNN)**
# One of the simplest ideas in ML — to classify a new point, just look at the K nearest training examples and take a majority vote.
# There's no real "training" — the model just memorizes the data. Simple but surprisingly effective for certain problems,
# though it gets slow and memory-heavy at scale.
#
# **So When Do You Use Neural Networks?**
# Neural networks and MLPs shine in specific scenarios — primarily when dealing with unstructured data like images, audio,
# text, and video, where the raw features (pixels, waveforms, characters) have complex spatial or sequential relationships
# that simpler models can't capture. Deep learning also shines when you have enormous amounts of data,
# because neural networks scale much better with data than traditional methods.
#
# But for structured/tabular data — the kind that lives in spreadsheets and databases — gradient boosting methods like
# XGBoost frequently outperform neural networks and are much faster to train and easier to tune.
# This is a common misconception that deep learning has "won everything," when in reality traditional ML methods are
# still dominant in many real-world business applications.
#
#
# **A Rough Mental Map**
# You can think of the tradeoffs like this: as you move from
#
# linear models → decision trees → ensembles → neural networks, you generally gain the ability to model more complex patterns,
#
# but you also lose interpretability, require more data, and need more compute. The right model is always the one that fits
# your data size, complexity, and constraints — not necessarily the most sophisticated one.
#
# ---
#
# ## **Techniques for Minimizing Error**
#
# This field is called optimization, and it's one of the most active areas of ML research.
# The goal is always the same: find the weight values that minimize the loss function.
#
# **The Foundation — Gradient Descent**
# Everything starts here. The gradient is essentially the slope of the loss function with respect to each weight —
# it tells you which direction the error increases. You move the weights in the opposite direction of the gradient (downhill),
# by a small step determined by the learning rate.
#
# The learning rate is critical. Too large and you overshoot the minimum and the model diverges. Too small and training takes forever or gets stuck.
#
# ---
#
# #### What Are the Internal Weights?
#
# Weights are simply numbers stored inside the model that determine how it transforms inputs into outputs.
# They have no meaning at the start — they're usually initialized randomly — and the entire training process is about
# finding the right values for them.
#
# In a neural network specifically, weights exist in a few forms:
#
# **Connection weights** — Every connection between two neurons has a weight.
# It's a multiplier that says "how much should this signal matter?" A high positive weight amplifies a signal,
# a near-zero weight ignores it, and a negative weight suppresses it.
#
# **Biases** — Each neuron also has a bias term, which is an offset that shifts the output up or down independent of the input.
# It gives the model more flexibility, similar to the intercept in a linear equation like y = mx + b.
#
# In other model types, the "weights" have different names but the same concept applies.
# In linear regression they're called coefficients. In decision trees they're the threshold values at each split.
# In SVMs they define the decision boundary. The universal idea is: these are the numbers the algorithm tunes to fit the data.
#
# At scale, a large neural network can have billions of these weights. GPT-style models are essentially just enormous
# collections of numbers that were tuned on massive datasets until the patterns in language emerged.
#
# ## **Techniques for Minimizing Error**
#
# This field is called optimization, and it's one of the most active areas of ML research.
# The goal is always the same: find the weight values that minimize the loss function.
#
# **The Foundation — Gradient Descent**
# Everything starts here. The gradient is essentially the slope of the loss function with respect to each weight —
# it tells you which direction the error increases. You move the weights in the opposite direction of the gradient (downhill),
# by a small step determined by the learning rate.
#
# The learning rate is critical. Too large and you overshoot the minimum and the model diverges. Too small and training takes forever or gets stuck.
#
#
# #### **There are three flavors of basic gradient descent:**
# **Batch Gradient Descent** — Compute the gradient over the entire dataset before updating weights.
# Very accurate but extremely slow and memory-heavy for large datasets.
#
# **Stochastic Gradient Descent (SGD)** — Update weights after every single training example.
# Much faster but very noisy — the loss bounces around a lot because each single example is a poor estimate of the true gradient.
#
# **Mini-batch Gradient Descent** — The practical standard. Split data into small batches (say 32 or 256 examples),
# compute gradient on each batch, update weights. Balances speed and stability. Almost all modern training uses this.
#
# ---
#
# #### **Advanced Optimizers (Improvements on SGD)**
#
# **Raw SGD has weaknesses** — it treats all weights equally and can be slow to converge. These optimizers address that:
#
# **Momentum** — Adds a "velocity" term to the weight updates. Instead of just following the current gradient,
# you accumulate a rolling average of past gradients. This helps the optimizer move faster through flat regions and resist
# getting stuck in small bumps. Think of a ball rolling downhill — it builds up speed rather than stopping at every dip.
#
# **RMSprop** — Adapts the learning rate for each weight individually based on how large its recent gradients have been.
# Weights that have been updated a lot get a smaller learning rate, and rarely updated weights get a larger one.
# This prevents any one weight from dominating training.
#
# **Adam (Adaptive Moment Estimation)** — Combines both Momentum and RMSprop. It tracks both the rolling average of
# gradients (like momentum) and the rolling average of squared gradients (like RMSprop) to give each weight its own
# adaptive learning rate.
# Adam is the default choice in most modern deep learning — it's robust and works well out of the box.
#
# **AdaGrad** — An earlier adaptive method that accumulates all past squared gradients.
# It works well for sparse data (like NLP) but has a problem: the learning rate shrinks continuously and eventually
# becomes so small the model stops learning. RMSprop was invented to fix this.
#
# **AdamW** — A variant of Adam that decouples weight decay (a regularization technique) from the gradient update.
# It's now the standard optimizer for training large language models.
#
# ---
#
# #### **Learning Rate Strategies**
#
# **The learning rate itself can be dynamic rather than fixed:**
#
# **Learning rate scheduling** — You start with a higher learning rate to explore broadly, then reduce it over time to fine-tune.
# Common schedules include step decay (reduce by half every N epochs) and cosine annealing (smoothly oscillate down).
#
# **Warmup** — Start with a very small learning rate, ramp it up over the first few thousand steps, then decay it.
# This is standard in transformer training because large random gradients early on can destabilize training.
#
# #### **Regularization — Preventing Overfitting**
#
# These aren't optimizers but they directly affect the error landscape by penalizing complexity:
#
# **L2 Regularization (Weight Decay)** — Adds a penalty to the loss proportional to the square of the weights.
# This discourages large weights and keeps the model simple, reducing overfitting.
#
# **L1 Regularization (Lasso)** — Penalizes the absolute value of weights. This tends to drive many weights all the way to zero,
# effectively removing connections and producing a sparse model.
#
# **Dropout** — During training, randomly "turn off" a percentage of neurons each forward pass. This forces the network to
# not rely too heavily on any single pathway, making it more robust. At inference time all neurons are active.
#
# **Early Stopping** — Monitor performance on the validation set during training.
# When validation loss stops improving (even if training loss keeps falling), stop training. Simple but surprisingly effective.
#
# ---
#
# #### Mathematical Explainer: How Weights Are Calculated
#
# ## The Mathematics of Weights: A Step-by-Step Walkthrough
#
# This section takes you through the *exact* mechanics of how a model learns — with real numbers at every step.
# No hand-waving. By the end, you will understand precisely what happens inside the model during training.
#
# ---
#
# ### Step 0 — Setting Up the Problem
#
# Let's say we want to train a model to predict house prices (in $100k) from a single feature: size in hundreds of square feet.
#
# We have one training example:
#
#     Input :  x = 3.0        (i.e. 300 sq ft)
#     Target:  y_true = 6.0   (i.e. $600,000)
#
# The true relationship we're trying to learn is: y = 2x
# (So the "perfect" weight would be w = 2.0, bias b = 0.0)
#
# But the model doesn't know that. It starts with random guesses.
#
# ---
#
# **Single Neuron**
#
# ### Step 1 — Initialise the Weights (Random Starting Point)
#
# Before training begins, weights are randomly initialised. Let's say:
#
#     w = 0.5    (random guess — nowhere near the true value of 2.0)
#     b = 0.0    (bias initialised to zero)
#
# These numbers mean nothing yet. Training is the process of correcting them.
#
# ---
#
# ### Step 2 — The Forward Pass (Making a Prediction)
#
# The model feeds the input through the equation to produce a prediction.
# For a single neuron with no activation function (linear), this is just:
#
#     y_pred = (w × x) + b
#
# Plugging in our values:
#
#     y_pred = (0.5 × 3.0) + 0.0
#     y_pred = 1.5
#
# The model predicted 1.5 ($150,000). The real answer is 6.0 ($600,000).
# That's a massive error. The model needs to correct its weights.
#
# ---
#
# ### Step 3 — The Loss Function (Measuring How Wrong We Are)
#
# The loss function converts the error into a single number the model can optimise.
# We'll use Mean Squared Error (MSE):
#
#     Loss = (y_pred − y_true)²
#
#     Loss = (1.5 − 6.0)²
#     Loss = (−4.5)²
#     Loss = 20.25
#
# The loss is 20.25. Our goal is to reduce this number as close to 0 as possible
# by adjusting w and b. The question is: which direction do we adjust them?
#
# ---
#
# ### Step 4 — The Gradient (Which Way is "Downhill"?)
#
# The gradient tells us the slope of the loss with respect to each weight —
# i.e. if we increase a weight by a tiny amount, does the loss go up or down, and by how much?
#
# We compute this using calculus (the chain rule). Breaking it into steps:
#
# **Gradient with respect to w:**
#
#     dLoss/dw = dLoss/dy_pred × dy_pred/dw
#
#     dLoss/dy_pred = 2 × (y_pred − y_true) = 2 × (1.5 − 6.0) = −9.0
#
#     dy_pred/dw = x = 3.0     (because y_pred = wx + b, so the rate of change w.r.t w is just x)
#
#     dLoss/dw = −9.0 × 3.0 = −27.0
#
# **Gradient with respect to b:**
#
#     dLoss/db = dLoss/dy_pred × dy_pred/db
#
#     dy_pred/db = 1.0          (because y_pred = wx + b, so the rate of change w.r.t b is just 1)
#
#     dLoss/db = −9.0 × 1.0 = −9.0
#
# What does a negative gradient mean?
# It means increasing w or b would decrease the loss. So to reduce error, we should increase them.
# This is exactly what the weight update rule does.
#
# ---
#
# ### Step 5 — The Weight Update (Gradient Descent)
#
# We update each weight by moving it a small step in the opposite direction of the gradient.
# The "small step" is controlled by the learning rate (lr) — a hyperparameter we set manually.
#
#     w_new = w − (lr × dLoss/dw)
#     b_new = b − (lr × dLoss/db)
#
# Let's use lr = 0.01:
#
#     w_new = 0.5 − (0.01 × −27.0) = 0.5 + 0.27  = 0.77
#     b_new = 0.0 − (0.01 × −9.0)  = 0.0 + 0.09  = 0.09
#
# After just one update:
#     w moved from 0.5 → 0.77   (getting closer to the true value of 2.0)
#     b moved from 0.0 → 0.09
#
# ---
#
# ### Step 6 — Iteration 2: Forward Pass Again
#
# We repeat the entire process with the updated weights.
#
#     y_pred = (0.77 × 3.0) + 0.09 = 2.31 + 0.09 = 2.40
#
#     Loss = (2.40 − 6.0)² = (−3.6)² = 12.96
#
# The loss dropped from 20.25 → 12.96. The model is already getting better.
#
#     dLoss/dy_pred = 2 × (2.40 − 6.0) = −7.2
#
#     dLoss/dw = −7.2 × 3.0 = −21.6
#     dLoss/db = −7.2 × 1.0 = −7.2
#
#     w_new = 0.77 − (0.01 × −21.6) = 0.77 + 0.216 = 0.986
#     b_new = 0.09 − (0.01 × −7.2)  = 0.09 + 0.072 = 0.162
#
# ---
#
# ### Step 7 — Watching the Weights Converge
#
# Here is how the weights and loss evolve across 10 iterations (lr = 0.01):
#
#     Iteration │    w     │    b     │   Loss
#     ──────────┼──────────┼──────────┼─────────
#         0     │  0.500   │  0.000   │  20.250
#         1     │  0.770   │  0.090   │  12.960
#         2     │  0.986   │  0.162   │   8.294
#         3     │  1.159   │  0.219   │   5.308
#         4     │  1.297   │  0.265   │   3.397
#         5     │  1.408   │  0.302   │   2.174
#         6     │  1.496   │  0.332   │   1.391
#         7     │  1.567   │  0.356   │   0.891
#         8     │  1.623   │  0.375   │   0.570
#         9     │  1.669   │  0.390   │   0.365
#        10     │  1.705   │  0.402   │   0.234
#
# The model started with w = 0.5 and is converging toward the true value of w = 2.0.
# Given enough iterations, it will get there. This is the entire engine of supervised learning.
#
# ---
#
# **Multi-Layer Backpropagation**
#
# ### Part 2 — Backpropagation Through a Multi-Layer Network
#
# The single neuron example shows the core idea. But real networks have multiple layers.
# Backpropagation is the algorithm that extends gradient descent through all of them — layer by layer,
# using the chain rule of calculus.
#
# Here's a tiny 3-layer network: 1 input → 1 hidden neuron (ReLU) → 1 output neuron (linear)
#
# **Setup:**
#
#     Input:       x = 2.0
#     Target:      y_true = 1.0
#
#     Hidden layer:   w1 = 0.5,   b1 = 0.1   (weights into the hidden neuron)
#     Output layer:   w2 = 0.3,   b2 = 0.1   (weights into the output neuron)
#
#     Activation on hidden neuron: ReLU(z) = max(0, z)
#     Loss: MSE
#
# ---
#
# #### Forward Pass (Left → Right)
#
# **Layer 1 — Hidden Neuron:**
#
#     z1    = (w1 × x) + b1
#     z1    = (0.5 × 2.0) + 0.1  =  1.1
#
#     h     = ReLU(z1)  =  max(0, 1.1)  =  1.1
#
#     (Since 1.1 > 0, ReLU passes it through unchanged.)
#
# **Layer 2 — Output Neuron:**
#
#     z2    = (w2 × h) + b2
#     z2    = (0.3 × 1.1) + 0.1  =  0.33 + 0.1  =  0.43
#
#     y_pred = 0.43    (no activation on output for regression)
#
# **Loss:**
#
#     Loss = (y_pred − y_true)²  =  (0.43 − 1.0)²  =  (−0.57)²  =  0.3249
#
# ---
#
# #### Backward Pass (Right → Left — Backpropagation)
#
# We now flow the error signal backwards through the network, layer by layer.
#
# **Step 1 — Gradient at the Output**
#
#     dLoss/dy_pred  =  2 × (y_pred − y_true)  =  2 × (−0.57)  =  −1.14
#
# **Step 2 — Gradients for Output Layer Weights (w2, b2)**
#
#     dLoss/dw2  =  dLoss/dy_pred × dy_pred/dw2  =  −1.14 × h     =  −1.14 × 1.1   =  −1.254
#     dLoss/db2  =  dLoss/dy_pred × dy_pred/db2  =  −1.14 × 1.0   =  −1.14
#
# **Step 3 — Pass the Error Signal Through to the Hidden Layer**
#
# The error signal needs to travel back through w2 and through the ReLU activation:
#
#     dLoss/dh  =  dLoss/dy_pred × dy_pred/dh  =  −1.14 × w2  =  −1.14 × 0.3  =  −0.342
#
# Now we pass through the ReLU derivative.
# ReLU's derivative is simple: it's 1 if the input was positive, 0 if it was negative (it "gates" the gradient).
#
#     dh/dz1  =  1   (because z1 = 1.1 > 0, ReLU was active)
#
#     dLoss/dz1  =  dLoss/dh × dh/dz1  =  −0.342 × 1  =  −0.342
#
# **Step 4 — Gradients for Hidden Layer Weights (w1, b1)**
#
#     dLoss/dw1  =  dLoss/dz1 × dz1/dw1  =  −0.342 × x   =  −0.342 × 2.0  =  −0.684
#     dLoss/db1  =  dLoss/dz1 × dz1/db1  =  −0.342 × 1.0  =  −0.342
#
# ---
#
# #### Weight Update (lr = 0.1)
#
#     w2_new  =  0.3   − (0.1 × −1.254)  =  0.3   + 0.1254  =  0.4254
#     b2_new  =  0.1   − (0.1 × −1.14)   =  0.1   + 0.114   =  0.2140
#     w1_new  =  0.5   − (0.1 × −0.684)  =  0.5   + 0.0684  =  0.5684
#     b1_new  =  0.1   − (0.1 × −0.342)  =  0.1   + 0.0342  =  0.1342
#
# ---
#
# #### Verification — Forward Pass with Updated Weights
#
#     z1     = (0.5684 × 2.0) + 0.1342  =  1.1368 + 0.1342  =  1.271
#     h      = ReLU(1.271) = 1.271
#
#     z2     = (0.4254 × 1.271) + 0.214  =  0.5407 + 0.214  =  0.7547
#     y_pred = 0.7547
#
#     Loss   = (0.7547 − 1.0)²  =  (−0.2453)²  =  0.0602
#
# The loss dropped from 0.3249 → 0.0602 in a single training step.
# A reduction of over 81%. The model is learning rapidly.
#
# ---
#
# ### The Key Intuitions to Hold Onto
#
# **1. The gradient is the model's compass.**
# It doesn't tell the model where the answer is — it just tells it which direction the error is increasing.
# The model always steps in the opposite direction.
#
# **2. The learning rate controls step size.**
# Too large: the model overshoots the minimum and bounces around or diverges.
# Too small: training takes forever or stalls.
# This is why tuning the learning rate is one of the most important skills in ML.
#
# **3. Backpropagation is just the chain rule applied systematically.**
# Each layer's gradient is computed by multiplying together all the local gradients from output back to input.
# This is how the error signal from the final loss reaches weights in the very first layer.
#
# **4. Activations like ReLU control whether a gradient flows.**
# When a ReLU neuron outputs zero (because its input was negative), its gradient is also zero.
# The error signal is completely blocked — that neuron contributed nothing to the output,
# so it receives no update. This is called a "dead neuron" when it happens persistently.
#
# **5. Every weight update is infinitesimally small by design.**
# Individual weight changes like 0.5 → 0.77 feel small. Across millions of examples and thousands of epochs,
# these tiny adjustments accumulate into a model that has genuinely learned the structure of the data.
#
# ---
#
# ### The Complete Training Loop in One View
#
#     INITIALISE weights randomly (w, b)
#
#     REPEAT for each training batch:
#     │
#     ├── FORWARD PASS
#     │     z = (w × x) + b
#     │     y_pred = activation(z)
#     │
#     ├── COMPUTE LOSS
#     │     loss = LossFunction(y_pred, y_true)
#     │
#     ├── BACKWARD PASS (Backpropagation)
#     │     Compute dLoss/dw and dLoss/db for every layer
#     │     using the chain rule, flowing right → left
#     │
#     └── UPDATE WEIGHTS
#           w = w − (lr × dLoss/dw)
#           b = b − (lr × dLoss/db)
#
#     UNTIL loss is acceptably small or validation performance stops improving
#
#
# #### What is Acceptably Small ?
#
# It entirely depends on the problem — there is no universal threshold.
#
# **Acceptably small** is one of those phrases that sounds precise but is actually very context-sensitive.
#
# Here's how to think about it properly:
#
# **The loss value itself is almost meaningless in isolation**
#
# The raw number only makes sense relative to the loss function being used and the scale of the data.
# For example, if you're predicting house prices in raw dollars (e.g. $450,000) using MSE, a loss of 500,000,000 might
# actually be fine — because MSE squares the error, so you're looking at squared dollars.
# If you're predicting a probability between 0 and 1 using binary cross-entropy, a loss of 0.15 is considered quite good.
# The same number means completely different things in different contexts.
#
#
# **What practitioners actually watch**
# Rather than checking if the loss crossed some magic threshold, experienced practitioners watch for two things.
#
# First, they monitor the direction — is the loss still decreasing, or has it plateaued?
# A loss that has genuinely stopped improving across many epochs is a signal to stop, regardless of its absolute value.
#
# Second, they translate the loss into a human-readable metric for the specific task — accuracy for classification,
# RMSE or MAE for regression — and ask whether that number is good enough for the real-world use case.
#
# So a self-driving car model and a movie recommendation model might both reach "acceptable" performance at completely
# different loss values, because what "good enough" means is defined by the application, not by mathematics.
#
# **Where 0.0 to 0.5 thinking comes from ? if you think loss should be between 0.0 and 0.5**
#
# You're likely thinking of this range because metrics like accuracy (0% to 100%) or
# cross-entropy loss on well-calibrated classification problems do tend to live in that neighborhood.
# But even then, a binary cross-entropy of 0.3 might be excellent for a hard medical diagnosis problem and terrible for
# a simple spam filter.
# The benchmark is always the baseline —
#
# **how well does a naive model (e.g. always predicting the majority class) do?**
#
# Your model needs to meaningfully beat that.
#
# **The practical answer for your module**
# The condition to stop training is really three things combined: the training loss has stopped meaningfully decreasing,
# the validation loss has stopped improving (or is starting to increase — which signals overfitting), and the real-world
# metric meets your acceptance criteria for the application. It's a human judgment call informed by those signals,
# not a mathematical absolute.
#
#
# """
#
# OPERATIONS = {
# }
#
# VISUAL_HTML = ""  # Add your HTML visual breakdown here
