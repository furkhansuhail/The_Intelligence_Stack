"""
Self-contained HTML visual for Supervised Learning Core Idea.
6 interactive tabs: Pipeline, Training Loop, Bias-Variance,
Optimizers, Model Landscape, Algorithm Zoo.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(SL_CORE_HTML, height=SL_CORE_HEIGHT, scrolling=True)
"""

SL_CORE_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
button{cursor:pointer;font-family:inherit;}
input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#a78bfa;cursor:pointer;}
table{border-collapse:separate;border-spacing:0 4px;width:100%;}
th{text-align:left;padding:6px 10px;color:#3f3f46;font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:1px;}
td{padding:7px 10px;font-size:9px;}
@keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes slideIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:none}}
.fade{animation:fadeIn .35s ease both;}
.card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
.tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
.tab-btn{padding:12px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:10px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
.tab-btn.active{border-bottom-color:#a78bfa;color:#a78bfa;}
.section-title{text-align:center;margin-bottom:20px;}
.section-title h2{font-size:18px;font-weight:800;margin-bottom:4px;color:#e4e4e7;}
.section-title p{font-size:11px;color:#71717a;}
.insight{max-width:1100px;margin:16px auto 0;padding:16px 22px;background:rgba(167,139,250,.06);border-radius:10px;border:1px solid rgba(167,139,250,.2);}
.ins-title{font-size:11px;font-weight:700;color:#a78bfa;margin-bottom:6px;}
.ins-body{font-size:11px;color:#71717a;line-height:1.8;}
canvas{display:block;}
</style>
</head>
<body>
<div id="app" style="max-width:1400px;margin:0 auto;padding:24px 16px;"></div>
<script>
/* ─── PALETTE ─── */
var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",
  accent:"#a78bfa",blue:"#4ecdc4",purple:"#c084fc",
  yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",
  dim:"#3f3f46",red:"#ef4444",green:"#4ade80",
  cyan:"#38bdf8",orange:"#fb923c",pink:"#f472b6"};

/* ─── STATE ─── */
var S={
  tab:0,
  pipeStep:-1,
  trainIter:0, trainPlay:false, trainTimer:null,
  bvComplexity:5,
  optSel:2,
  mlType:0, mlModel:0,
  algoCategory:0
};

/* ─── HELPERS ─── */
function hex(c,a){var r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);return 'rgba('+r+','+g+','+b+','+a+')';}
function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
function span(st,inner){return '<span style="'+st+'">'+inner+'</span>';}
function card(inner,extraStyle){return '<div class="card" style="max-width:1100px;margin:0 auto 14px;'+(extraStyle||'')+'">'+inner+'</div>';}
function sectionTitle(t,s){return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';}
function insight(icon,title,body){return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';}
function chip(text,color,small){var fs=small?'7px':'9px';return '<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:'+fs+';font-weight:700;background:'+hex(color,.15)+';border:1px solid '+hex(color,.4)+';color:'+color+';">'+text+'</span>';}
function statBox(label,value,color,sub){
  return '<div style="text-align:center;min-width:90px;">'
    +'<div style="font-size:8px;color:'+C.muted+';margin-bottom:4px;letter-spacing:1px;">'+label+'</div>'
    +'<div style="font-size:20px;font-weight:800;color:'+color+';">'+value+'</div>'
    +(sub?'<div style="font-size:7px;color:'+C.dim+';margin-top:2px;">'+sub+'</div>':'')
    +'</div>';}
function btnSel(idx,cur,color,label,action){
  var on=idx===cur;
  return '<button data-action="'+action+'" data-idx="'+idx+'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';}

/* ═══════════════════════════════════════════════════════
   TAB 1 — THE PIPELINE
═══════════════════════════════════════════════════════ */
function renderPipeline(){
  var stages=[
    {label:"Collect Data",     icon:"&#128190;", color:C.muted,   sub:"Labeled examples"},
    {label:"Extract Features", icon:"&#128268;", color:C.blue,    sub:"Input representation"},
    {label:"Choose Model",     icon:"&#129302;", color:C.cyan,    sub:"Architecture"},
    {label:"Forward Pass",     icon:"&#8594;",   color:C.yellow,  sub:"Make prediction"},
    {label:"Compute Loss",     icon:"&#128200;", color:C.orange,  sub:"Measure error"},
    {label:"Backpropagate",    icon:"&#8592;",   color:C.purple,  sub:"Flow gradients"},
    {label:"Update Weights",   icon:"&#9881;",   color:C.accent,  sub:"Gradient descent"},
    {label:"Validate & Deploy",icon:"&#128640;", color:C.green,   sub:"Generalize"},
  ];
  var details=[
    {title:"Collect Labeled Data",body:'Each example has an input x (features) and a known output y (label). E.g. thousands of emails already tagged spam/not-spam. This is what defines supervised learning — the supervisor provides correct answers.',color:C.muted},
    {title:"Extract Features",body:'Raw data is converted into numerical form the model can use. Images become pixel arrays, text becomes token IDs or embeddings, tabular data becomes float vectors. Feature engineering can dramatically affect model performance.',color:C.blue},
    {title:"Choose Model Architecture",body:'Select the type of function to learn: linear models, decision trees, SVMs, neural networks, ensembles. The choice depends on data size, problem complexity, and whether you need interpretability.',color:C.cyan},
    {title:"Forward Pass — Make a Prediction",body:'Input x flows through the model: y_pred = f(x; w, b). For a single neuron: y_pred = (w &times; x) + b. For a deep network, this is repeated across dozens of layers. The result is the model\'s current best guess.',color:C.yellow},
    {title:"Compute Loss — Measure the Error",body:'Compare y_pred to y_true using a loss function. MSE for regression: L = (y_pred &minus; y_true)&sup2;. Cross-entropy for classification. The raw loss value is what the optimizer will minimize.',color:C.orange},
    {title:"Backpropagation — Flow Gradients Backward",body:'Using the chain rule of calculus, compute dL/dw for every weight in the network. Error signal flows right-to-left through each layer. Each weight learns how much it contributed to the mistake.',color:C.purple},
    {title:"Update Weights — Gradient Descent",body:'w_new = w &minus; (lr &times; dL/dw). Each weight moves a small step in the direction that reduces loss. The learning rate controls step size. This repeats thousands of times across your training data.',color:C.accent},
    {title:"Validate, Test & Deploy",body:'Evaluate on a held-out validation set to catch overfitting. Final evaluation on the test set gives unbiased performance. If metrics meet requirements, deploy to production and monitor for distribution shift.',color:C.green},
  ];
  var paradigms=[
    {name:"Supervised",color:C.accent,def:"Labels provided. Learn input &rarr; output mapping.",eg:"Spam detection, image classification, price prediction"},
    {name:"Unsupervised",color:C.blue,def:"No labels. Find hidden structure in data.",eg:"Clustering, dimensionality reduction, anomaly detection"},
    {name:"Reinforcement",color:C.yellow,def:"Agent learns via reward/penalty from environment.",eg:"Game playing, robotics, recommendation systems"},
  ];
  var step=S.pipeStep;
  var out=sectionTitle("The Supervised Learning Pipeline","From raw labeled data to a deployed model — every step, every time");

  // Pipeline stages
  out+=card(
    '<div style="overflow-x:auto;"><div style="display:flex;gap:0;align-items:center;min-width:800px;padding:12px 0 8px;">'
    +stages.map(function(s,i){
      var active=step===i;
      return '<button data-action="pipeStep" data-idx="'+i+'" style="flex:1;min-width:80px;padding:10px 4px;border-radius:8px;border:1.5px solid '+(active?s.color:hex(s.color,.3))+';background:'+(active?hex(s.color,.12):'#0d0d14')+';color:'+(active?s.color:hex(s.color,.7))+';font-size:8px;font-weight:700;font-family:inherit;cursor:pointer;line-height:1.6;transition:all .3s;'+(active?'box-shadow:0 0 12px '+hex(s.color,.3)+';':'')+'">'
        +s.icon+'<br>'+s.label+'<br><span style="font-size:7px;color:'+C.dim+';font-weight:400;">'+s.sub+'</span>'
        +'</button>'
        +(i<stages.length-1?'<div style="color:'+C.dim+';font-size:14px;padding:0 2px;flex-shrink:0;">&#8250;</div>':'');
    }).join('')
    +'</div></div>'
    +(step===-1
      ? div('text-align:center;padding:16px 0;color:'+C.dim+';font-size:9px;','&#9654; Click any stage to inspect it')
      : '<div class="fade" style="margin-top:12px;padding:14px 16px;border-radius:8px;background:'+hex(details[step].color,.07)+';border:1px solid '+hex(details[step].color,.25)+'">'
        +div('font-size:12px;font-weight:700;color:'+details[step].color+';margin-bottom:6px;',details[step].title)
        +div('font-size:10px;color:'+C.muted+';line-height:1.7;',details[step].body)
        +'</div>')
  );

  // Three paradigms
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','The Three Learning Paradigms')
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +paradigms.map(function(p){
      return '<div style="flex:1;min-width:220px;padding:12px 16px;border-radius:8px;background:'+hex(p.color,.07)+';border:1px solid '+hex(p.color,.25)+'">'
        +div('font-size:13px;font-weight:800;color:'+p.color+';margin-bottom:6px;',p.name+' Learning')
        +div('font-size:9px;color:'+C.muted+';line-height:1.8;margin-bottom:4px;',p.def)
        +div('font-size:8px;color:'+C.dim+';font-style:italic;','e.g. '+p.eg)
        +'</div>';
    }).join('')
    +'</div>'
  );

  // ERM
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Empirical Risk Minimization (ERM) — The Formal Objective')
    +'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
    +'<div style="flex:1;min-width:240px;padding:12px 16px;border-radius:8px;background:'+hex(C.red,.07)+';border:1px solid '+hex(C.red,.2)+'">'
    +div('font-size:10px;font-weight:700;color:'+C.red+';margin-bottom:6px;','True Objective (Unknown)')
    +'<div style="font-family:inherit;font-size:10px;color:'+C.muted+';line-height:1.9;">Expected Risk = E[L(f(x; &theta;), y)]<br>'
    +span('font-size:8px;color:'+C.dim+';','We don\'t know the true distribution')+'</div>'
    +'</div>'
    +'<div style="flex:1;min-width:240px;padding:12px 16px;border-radius:8px;background:'+hex(C.green,.07)+';border:1px solid '+hex(C.green,.2)+'">'
    +div('font-size:10px;font-weight:700;color:'+C.green+';margin-bottom:6px;','Practical Approximation (Trainable)')
    +'<div style="font-family:inherit;font-size:10px;color:'+C.muted+';line-height:1.9;">Empirical Risk = (1/N) &Sigma; L(f(x&#7522;; &theta;), y&#7522;)<br>'
    +span('font-size:8px;color:'+C.dim+';','Average loss over training dataset')+'</div>'
    +'</div>'
    +'<div style="flex:1;min-width:200px;padding:12px 16px;border-radius:8px;background:'+hex(C.accent,.07)+';border:1px solid '+hex(C.accent,.2)+'">'
    +div('font-size:10px;font-weight:700;color:'+C.accent+';margin-bottom:6px;','Data Splits')
    +['Training — model learns from this','Validation — tune hyperparameters, catch overfitting early','Test — final blind evaluation (never touched during training)']
      .map(function(s,i){return div('font-size:8px;color:'+C.muted+';line-height:1.9;',(i===0?'&#9679; ':i===1?'&#9679; ':'&#9679; ')+s);}).join('')
    +'</div>'
    +'</div>'
  );

  out+=insight('&#128161;','The Fundamental Assumption — IID',
    'Supervised learning assumes training and deployment data come from the <strong style="color:'+C.accent+'">same distribution</strong> (IID: Independent, Identically Distributed). '
    +'When this breaks — a model trained pre-2020 deployed during a recession, a medical classifier trained at one hospital used at another — '
    +'performance can collapse even if test metrics looked perfect. '
    +'<strong style="color:'+C.yellow+'">Always ask: does my training data reflect deployment conditions?</strong>');
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 2 — TRAINING LOOP
═══════════════════════════════════════════════════════ */
function renderTraining(){
  // Pre-computed iterations: x=3, y_true=6, lr=0.01, start w=0.5 b=0
  var iters=[
    {w:0.500,b:0.000,loss:20.250},
    {w:0.770,b:0.090,loss:12.960},
    {w:0.986,b:0.162,loss:8.294},
    {w:1.159,b:0.219,loss:5.308},
    {w:1.297,b:0.265,loss:3.397},
    {w:1.408,b:0.302,loss:2.174},
    {w:1.496,b:0.332,loss:1.391},
    {w:1.567,b:0.356,loss:0.891},
    {w:1.623,b:0.375,loss:0.570},
    {w:1.669,b:0.390,loss:0.365},
    {w:1.705,b:0.402,loss:0.234},
  ];
  var n=S.trainIter;
  var cur=iters[n];
  var pct=Math.round((1-cur.loss/20.25)*100);
  var ypred=(cur.w*3+cur.b).toFixed(3);

  var loopSteps=[
    {label:"Forward Pass",color:C.yellow,icon:"&#8594;",
     eq:"y_pred = (w &times; x) + b = ("+cur.w.toFixed(3)+" &times; 3.0) + "+cur.b.toFixed(3)+" = "+ypred,
     note:"Model feeds input through current weights to produce a prediction"},
    {label:"Compute Loss",color:C.orange,icon:"&#128200;",
     eq:"Loss = (y_pred &minus; y_true)&sup2; = ("+ypred+" &minus; 6.0)&sup2; = "+cur.loss.toFixed(4),
     note:"MSE measures how far the prediction is from the true value"},
    {label:"Backpropagation",color:C.purple,icon:"&#8592;",
     eq:"dL/dw = 2(y_pred&minus;y_true)&times;x = "+((2*(parseFloat(ypred)-6)*3).toFixed(3))+"&nbsp;&nbsp;dL/db = "+((2*(parseFloat(ypred)-6)).toFixed(3)),
     note:"Chain rule computes gradient of loss w.r.t. every weight"},
    {label:"Weight Update",color:C.accent,icon:"&#9881;",
     eq:"w_new = w &minus; (0.01 &times; dL/dw)&nbsp;&nbsp;b_new = b &minus; (0.01 &times; dL/db)",
     note:"Gradient descent nudges weights downhill by a small step (lr=0.01)"},
  ];

  var out=sectionTitle("The Training Loop — Interactive Walkthrough","Watch a single neuron learn y = 2x with real numbers at every step");

  // Controls
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Single Neuron — Learning y = 2x &nbsp; (x=3.0, y_true=6.0, lr=0.01)')
    +'<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:16px;">'
    +'<div style="font-size:9px;color:'+C.muted+';min-width:80px;">Iteration: '+n+'/10</div>'
    +'<input type="range" min="0" max="10" value="'+n+'" data-action="trainIter" style="flex:1;min-width:140px;accent-color:'+C.accent+';">'
    +'<button data-action="trainBack" style="padding:6px 14px;border-radius:6px;border:1px solid '+C.border+';background:'+C.card+';color:'+C.muted+';font-size:11px;font-family:inherit;">&#9664;</button>'
    +'<button data-action="trainFwd" style="padding:6px 14px;border-radius:6px;border:1px solid '+C.border+';background:'+C.card+';color:'+C.muted+';font-size:11px;font-family:inherit;">&#9654;</button>'
    +'<button data-action="trainReset" style="padding:6px 14px;border-radius:6px;border:1px solid '+C.dim+';background:'+C.card+';color:'+C.dim+';font-size:9px;font-family:inherit;">Reset</button>'
    +'</div>'
    // Stats row
    +'<div style="display:flex;gap:20px;flex-wrap:wrap;justify-content:center;margin-bottom:16px;">'
    +statBox('w (weight)',cur.w.toFixed(3),(cur.w>1.8?C.green:cur.w>1.0?C.yellow:C.red),'target: 2.000')
    +statBox('b (bias)',cur.b.toFixed(3),C.blue,'target: 0.000')
    +statBox('Loss',cur.loss.toFixed(3),(cur.loss<1?C.green:cur.loss<5?C.yellow:C.orange),'MSE')
    +statBox('y_pred',ypred,(Math.abs(parseFloat(ypred)-6)<0.5?C.green:C.orange),'y_true: 6.000')
    +statBox('Error %',pct+'%',(pct>80?C.green:pct>50?C.yellow:C.red),'reduction')
    +'</div>'
    // Progress bar
    +'<div style="height:8px;background:#1e1e2e;border-radius:4px;overflow:hidden;margin-bottom:8px;">'
    +'<div style="height:100%;width:'+pct+'%;background:linear-gradient(90deg,'+C.red+','+C.yellow+','+C.green+');border-radius:4px;transition:width .4s;"></div>'
    +'</div>'
    +div('font-size:8px;color:'+C.dim+';text-align:center;','Loss reduction: '+pct+'% | w converging toward 2.0')
  );

  // Loop steps
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:14px;','Training Loop Steps — Iteration '+n)
    +'<div style="display:flex;flex-direction:column;gap:8px;">'
    +loopSteps.map(function(s,i){
      return '<div class="fade" style="display:flex;gap:12px;align-items:flex-start;padding:12px 14px;border-radius:8px;background:'+hex(s.color,.07)+';border:1px solid '+hex(s.color,.25)+';animation-delay:'+(i*0.07)+'s;">'
        +'<div style="font-size:16px;width:24px;flex-shrink:0;text-align:center;">'+s.icon+'</div>'
        +'<div style="flex:1;">'
        +div('font-size:10px;font-weight:700;color:'+s.color+';margin-bottom:4px;',s.label)
        +div('font-family:inherit;font-size:9px;color:'+C.muted+';padding:6px 10px;border-radius:5px;background:'+hex(s.color,.05)+';margin-bottom:4px;',s.eq)
        +div('font-size:8px;color:'+C.dim+';',s.note)
        +'</div></div>';
    }).join('')
    +'</div>'
  );

  // Convergence table
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Weight Convergence — All 10 Iterations')
    +'<div style="overflow-x:auto;"><table><thead><tr>'
    +['Iter','w','b','Loss','y_pred','Status'].map(function(h){return '<th>'+h+'</th>';}).join('')
    +'</tr></thead><tbody>'
    +iters.map(function(it,i){
      var pred=(it.w*3+it.b).toFixed(3);
      var hl=i===n;
      var bg=hl?hex(C.accent,.12):'transparent';
      var status=it.loss<0.5?chip('Converged',C.green,true):it.loss<5?chip('Learning',C.yellow,true):chip('Far',C.red,true);
      return '<tr style="background:'+bg+';">'
        +'<td style="color:'+(hl?C.accent:C.muted)+';font-weight:'+(hl?'800':'400')+';">'+i+'</td>'
        +'<td style="color:'+C.blue+';">'+it.w.toFixed(3)+'</td>'
        +'<td style="color:'+C.cyan+';">'+it.b.toFixed(3)+'</td>'
        +'<td style="color:'+(it.loss<1?C.green:it.loss<5?C.yellow:C.orange)+';font-weight:700;">'+it.loss.toFixed(3)+'</td>'
        +'<td style="color:'+C.muted+';">'+pred+'</td>'
        +'<td>'+status+'</td>'
        +'</tr>';
    }).join('')
    +'</tbody></table></div>'
  );

  // Backprop multi-layer
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Multi-Layer Backpropagation — 3-Layer Network')
    +'<div style="font-size:9px;color:'+C.muted+';line-height:1.8;margin-bottom:12px;">'
    +'Setup: x=2.0, y_true=1.0 &nbsp;|&nbsp; Hidden: w1=0.5, b1=0.1, ReLU &nbsp;|&nbsp; Output: w2=0.3, b2=0.1, linear &nbsp;|&nbsp; lr=0.1'
    +'</div>'
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +[
      {dir:"Forward &#8594;",color:C.yellow,steps:[
        "z1 = (0.5&times;2.0)+0.1 = 1.1",
        "h = ReLU(1.1) = 1.1",
        "z2 = (0.3&times;1.1)+0.1 = 0.43",
        "Loss = (0.43&minus;1.0)&sup2; = 0.3249"
      ]},
      {dir:"Backward &#8592;",color:C.purple,steps:[
        "dL/dy = 2&times;(0.43&minus;1.0) = &minus;1.14",
        "dL/dw2 = &minus;1.14&times;1.1 = &minus;1.254",
        "dL/dh = &minus;1.14&times;0.3 = &minus;0.342",
        "dL/dw1 = &minus;0.342&times;2.0 = &minus;0.684"
      ]},
      {dir:"After Update &#9881;",color:C.green,steps:[
        "w2: 0.3 &rarr; 0.4254",
        "w1: 0.5 &rarr; 0.5684",
        "y_pred: 0.43 &rarr; 0.7547",
        "Loss: 0.3249 &rarr; 0.0602 (&minus;81%)"
      ]},
    ].map(function(col){
      return '<div style="flex:1;min-width:200px;padding:12px 14px;border-radius:8px;background:'+hex(col.color,.07)+';border:1px solid '+hex(col.color,.25)+'">'
        +div('font-size:10px;font-weight:700;color:'+col.color+';margin-bottom:8px;',col.dir)
        +col.steps.map(function(s){return div('font-size:9px;color:'+C.muted+';line-height:1.9;font-family:inherit;',s);}).join('')
        +'</div>';
    }).join('')
    +'</div>'
  );

  out+=insight('&#128161;','The Five Intuitions to Hold Onto',
    '(1) <strong style="color:'+C.yellow+'">Gradient = compass</strong> — tells which direction loss increases; model steps opposite. &nbsp;'
    +'(2) <strong style="color:'+C.orange+'">Learning rate = step size</strong> — too large overshoots, too small stalls. &nbsp;'
    +'(3) <strong style="color:'+C.purple+'">Backprop = chain rule</strong> — error flows backward layer by layer. &nbsp;'
    +'(4) <strong style="color:'+C.red+'">Dead ReLU</strong> — if a ReLU neuron always outputs 0, its gradient is 0 and it never updates. &nbsp;'
    +'(5) <strong style="color:'+C.green+'">Small steps compound</strong> — 0.5&rarr;0.77 feels tiny; millions of updates produce a model that learned genuine structure.');
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 3 — BIAS-VARIANCE
═══════════════════════════════════════════════════════ */
function renderBiasVariance(){
  var c=S.bvComplexity; // 1-10
  // Simulated train/test error curves
  var trainErr=Math.max(0.05, 0.9*Math.pow(0.72,c));
  var testErr=c<=5 ? 0.9*Math.pow(0.72,c)+0.05 : 0.18+0.08*(c-5)*(c-5)*0.15;
  var regime=c<=3?"Underfitting (High Bias)":c<=6?"Good Fit (Sweet Spot)":"Overfitting (High Variance)";
  var regColor=c<=3?C.red:c<=6?C.green:C.orange;
  var gap=testErr-trainErr;
  var trainPct=Math.round(trainErr*100);
  var testPct=Math.round(testErr*100);

  var out=sectionTitle("Bias-Variance Tradeoff","The single most important concept for diagnosing model behaviour");

  // Complexity slider + live metrics
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Model Complexity Explorer')
    +'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px;">'
    +div('font-size:9px;color:'+C.muted+';min-width:130px;','Complexity: '+c+'/10')
    +'<input type="range" min="1" max="10" value="'+c+'" data-action="bvComplexity" style="flex:1;accent-color:'+regColor+';">'
    +'<div style="font-size:9px;color:'+regColor+';font-weight:700;min-width:220px;text-align:right;">'+regime+'</div>'
    +'</div>'
    +'<div style="display:flex;gap:20px;flex-wrap:wrap;justify-content:center;margin-bottom:16px;">'
    +statBox('Training Error',trainPct+'%',(trainPct<15?C.green:trainPct<40?C.yellow:C.red),'on training set')
    +statBox('Test Error',testPct+'%',(testPct<20?C.green:testPct<40?C.yellow:C.red),'on held-out set')
    +statBox('Generalization Gap',(Math.round(gap*100))+'%',(gap<0.08?C.green:gap<0.20?C.yellow:C.red),'test - train')
    +statBox('Regime',c<=3?'Bias':c<=6?'Good':' Var',regColor,'')
    +'</div>'
    // Visual bar chart
    +'<div style="display:flex;gap:16px;align-items:flex-end;height:80px;padding:0 20px;">'
    +'<div style="flex:1;display:flex;flex-direction:column;align-items:center;">'
    +'<div style="width:100%;max-width:80px;background:'+hex(C.blue,.7)+';border-radius:4px 4px 0 0;height:'+(trainPct*0.7)+'px;transition:height .4s;"></div>'
    +div('font-size:8px;color:'+C.muted+';margin-top:4px;','Train')
    +'</div>'
    +'<div style="flex:1;display:flex;flex-direction:column;align-items:center;">'
    +'<div style="width:100%;max-width:80px;background:'+hex(C.orange,.7)+';border-radius:4px 4px 0 0;height:'+(testPct*0.7)+'px;transition:height .4s;"></div>'
    +div('font-size:8px;color:'+C.muted+';margin-top:4px;','Test')
    +'</div>'
    +'<div style="flex:3;padding:0 10px;">'
    +div('font-size:9px;line-height:1.9;color:'+C.muted+';',
      (c<=3
        ?"<strong style='color:"+C.red+"'>High Bias (Underfitting)</strong><br>Model too simple. Misses real patterns.<br>Both train AND test error are high.<br>Fix: more complexity, better features."
        :c<=6
        ?"<strong style='color:"+C.green+"'>Sweet Spot (Good Generalization)</strong><br>Model captures real structure, ignores noise.<br>Train and test error are both low and close.<br>This is the goal."
        :"<strong style='color:"+C.orange+"'>High Variance (Overfitting)</strong><br>Model memorized training noise.<br>Train error low but test error high — large gap.<br>Fix: regularization, more data, simpler model."))
    +'</div>'
    +'</div>'
  );

  // Three scenarios
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Three Scenarios — Concrete Examples')
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +[
      {title:"High Bias — Underfitting",color:C.red,trainE:"HIGH",testE:"HIGH",
       eg:"Linear model on curved data. A straight line can't fit a parabola no matter how long you train.",
       fix:"Increase complexity. Add features. Use polynomial/non-linear model."},
      {title:"Optimal — Good Fit",color:C.green,trainE:"LOW",testE:"LOW",
       eg:"Random Forest on a tabular problem. Captures non-linear patterns, generalizes well.",
       fix:"This is the goal. Monitor for drift in production."},
      {title:"High Variance — Overfitting",color:C.orange,trainE:"LOW",testE:"HIGH",
       eg:"50-layer network on 200 training examples. Memorized every example including noise.",
       fix:"L1/L2 regularization. Dropout. More training data. Early stopping."},
    ].map(function(s){
      return '<div style="flex:1;min-width:220px;padding:12px 14px;border-radius:8px;background:'+hex(s.color,.07)+';border:1px solid '+hex(s.color,.25)+'">'
        +div('font-size:11px;font-weight:800;color:'+s.color+';margin-bottom:6px;',s.title)
        +'<div style="display:flex;gap:10px;margin-bottom:8px;">'
        +'<div style="padding:4px 10px;border-radius:4px;font-size:8px;font-weight:700;background:'+hex(C.blue,.12)+';color:'+C.blue+';">Train: '+s.trainE+'</div>'
        +'<div style="padding:4px 10px;border-radius:4px;font-size:8px;font-weight:700;background:'+hex(C.orange,.12)+';color:'+C.orange+';">Test: '+s.testE+'</div>'
        +'</div>'
        +div('font-size:9px;color:'+C.muted+';line-height:1.7;margin-bottom:6px;',s.eg)
        +div('font-size:8px;color:'+C.dim+';font-style:italic;','Fix: '+s.fix)
        +'</div>';
    }).join('')
    +'</div>'
  );

  // Regularization tools
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Regularization Tools — Controlling the Tradeoff')
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +[
      {name:"L2 (Ridge)",color:C.accent,desc:"Penalizes large weights by adding &lambda;&Sigma;w&sup2; to loss. Discourages complexity without zeroing weights."},
      {name:"L1 (Lasso)",color:C.blue,desc:"Penalizes |w|. Drives many weights to exactly zero — sparse model. Useful for feature selection."},
      {name:"Dropout",color:C.purple,desc:"Randomly deactivates neurons each forward pass. Forces redundant learning — no single pathway dominates."},
      {name:"Early Stopping",color:C.green,desc:"Stop training when validation loss stops improving. Simple, effective, free — always use it."},
      {name:"More Data",color:C.yellow,desc:"The best regularizer. More examples give the model less opportunity to memorize any single one."},
    ].map(function(r){
      return '<div style="flex:1;min-width:170px;padding:10px 14px;border-radius:8px;background:'+hex(r.color,.06)+';border:1px solid '+hex(r.color,.2)+'">'
        +div('font-size:10px;font-weight:700;color:'+r.color+';margin-bottom:6px;',r.name)
        +div('font-size:8px;color:'+C.muted+';line-height:1.7;',r.desc)
        +'</div>';
    }).join('')
    +'</div>'
  );

  out+=insight('&#9878;','Loss vs Metric — Two Different Things',
    '<strong style="color:'+C.accent+'">Loss</strong> is what the model optimizes — chosen for math properties (differentiability). E.g. Cross-Entropy, MSE. '
    +'<strong style="color:'+C.blue+'">Metric</strong> is how you evaluate for the real-world task. E.g. Accuracy, F1, AUC-ROC, RMSE. '
    +'A fraud detection model trains with Cross-Entropy (differentiable) but is evaluated on F1+AUC (because 99.9% of transactions are legitimate — accuracy is meaningless). '
    +'<strong style="color:'+C.yellow+'">The loss going down is a good sign. The metric on the test set is the truth.</strong>');
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 4 — OPTIMIZERS
═══════════════════════════════════════════════════════ */
function renderOptimizers(){
  var gdTypes=[
    {name:"Batch GD",color:C.muted,
     desc:"Compute gradient over the entire dataset before each update. Accurate gradient, but extremely slow and memory-heavy for large datasets.",
     pros:["Most accurate gradient estimate","Deterministic — same result each run"],
     cons:["Impractical for large datasets","Memory: entire dataset in RAM"]},
    {name:"Stochastic (SGD)",color:C.blue,
     desc:"Update weights after every single training example. Fast per-step but very noisy — loss bounces around because one example is a poor gradient estimate.",
     pros:["Very fast iterations","Can escape local minima due to noise"],
     cons:["Noisy — loss fluctuates wildly","Needs careful learning rate tuning"]},
    {name:"Mini-batch GD",color:C.green,
     desc:"Split data into small batches (32–256 examples), compute gradient per batch. The practical standard — balances accuracy and speed. This is what 'SGD' usually means in practice.",
     pros:["Practical standard for modern training","GPU-friendly parallelism","Good balance: speed + stability"],
     cons:["Batch size is another hyperparameter"]},
  ];
  var advOpts=[
    {name:"Momentum",color:C.cyan,year:"1964",
     formula:"v = &beta;v &minus; lr&times;g &nbsp;&nbsp; w = w + v",
     desc:"Accumulates a rolling average of past gradients. Builds speed through flat regions and resists small bumps. Think of a ball rolling downhill gaining momentum.",
     usedBy:"Default SGD + momentum in PyTorch"},
    {name:"RMSprop",color:C.yellow,year:"2012",
     formula:"s = &beta;s + (1&minus;&beta;)g&sup2; &nbsp;&nbsp; w -= lr&times;g / &radic;s",
     desc:"Adapts the learning rate per-weight based on recent gradient magnitude. Large gradients get smaller steps; rare gradients get larger steps.",
     usedBy:"RNNs, deep networks"},
    {name:"Adam",color:C.accent,year:"2014",
     formula:"m = &beta;&#8321;m + (1&minus;&beta;&#8321;)g &nbsp; v = &beta;&#8322;v + (1&minus;&beta;&#8322;)g&sup2; &nbsp; w -= lr&times;m&#770;/(&radic;v&#770;+&epsilon;)",
     desc:"Combines Momentum + RMSprop. Tracks both rolling mean of gradients AND rolling mean of squared gradients. Each weight gets its own adaptive learning rate. Robust out of the box — the default choice for deep learning.",
     usedBy:"Default for most deep learning"},
    {name:"AdamW",color:C.purple,year:"2017",
     formula:"Same as Adam + decoupled weight decay: w *= (1&minus;lr&times;&lambda;)",
     desc:"Adam with decoupled weight decay — L2 regularization is applied to the weights directly rather than through the gradient. Standard for training large language models.",
     usedBy:"GPT, BERT, all modern LLMs"},
    {name:"AdaGrad",color:C.orange,year:"2011",
     formula:"G = G + g&sup2; &nbsp;&nbsp; w -= lr&times;g / &radic;G",
     desc:"Accumulates ALL squared gradients. Good for sparse data (NLP). Problem: learning rate shrinks monotonically and eventually stops learning entirely. RMSprop was invented to fix this.",
     usedBy:"Sparse features, NLP (legacy)"},
  ];
  var sel=S.optSel;
  var v=advOpts[sel];

  var out=sectionTitle("Optimization — How Models Actually Learn","Gradient descent and its descendants — from basic to production-grade");

  // GD types
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Three Flavors of Gradient Descent')
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +gdTypes.map(function(g){
      return '<div style="flex:1;min-width:220px;padding:12px 14px;border-radius:8px;background:'+hex(g.color,.07)+';border:1px solid '+hex(g.color,.25)+'">'
        +div('font-size:12px;font-weight:800;color:'+g.color+';margin-bottom:6px;',g.name)
        +div('font-size:9px;color:'+C.muted+';line-height:1.7;margin-bottom:8px;',g.desc)
        +div('font-size:8px;color:'+C.green+';margin-bottom:4px;','&#10003; '+g.pros.join('<br>&#10003; '))
        +div('font-size:8px;color:'+C.red+';margin-top:6px;','&#8722; '+g.cons.join('<br>&#8722; '))
        +'</div>';
    }).join('')
    +'</div>'
  );

  // Advanced optimizer selector
  out+='<div class="card" style="max-width:1100px;margin:0 auto 14px;">';
  out+=div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Advanced Optimizers — Improvements on SGD');
  out+='<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:16px;">';
  advOpts.forEach(function(o,i){ out+=btnSel(i,sel,o.color,o.name+' ('+o.year+')','optSel'); });
  out+='</div>';
  out+='<div class="fade" style="display:flex;gap:16px;flex-wrap:wrap;padding:16px;border-radius:8px;background:'+hex(v.color,.07)+';border:1px solid '+hex(v.color,.25)+'">';
  out+='<div style="flex:2;min-width:260px;">';
  out+=div('font-size:15px;font-weight:800;color:'+v.color+';margin-bottom:8px;',v.name);
  out+=div('font-size:10px;color:'+C.muted+';line-height:1.7;margin-bottom:10px;',v.desc);
  out+='<div style="padding:8px 12px;border-radius:6px;background:'+hex(v.color,.08)+';border:1px solid '+hex(v.color,.3)+';font-family:inherit;font-size:9px;color:'+v.color+';white-space:pre-wrap;">'+v.formula+'</div>';
  out+='</div>';
  out+='<div style="min-width:160px;">';
  out+=div('font-size:8px;color:'+C.dim+';margin-bottom:4px;','Used by:');
  out+=div('font-size:9px;color:'+v.color+';font-weight:700;',v.usedBy);
  out+='</div>';
  out+='</div>';
  out+='</div>';

  // Comparison table
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Comparison Table')
    +'<div style="overflow-x:auto;"><table><thead><tr>'
    +['Optimizer','Adaptive LR','Momentum','Recommended For','Default?'].map(function(h){return '<th>'+h+'</th>';}).join('')
    +'</tr></thead><tbody>'
    +[
      {name:'SGD + Momentum',adaptive:'No',momentum:'Yes',rec:'CV, when you want control',def:'PyTorch CV',color:C.blue},
      {name:'RMSprop',adaptive:'Yes',momentum:'No',rec:'RNNs, recurrent models',def:'Keras default',color:C.yellow},
      {name:'Adam',adaptive:'Yes',momentum:'Yes',rec:'Most deep learning',def:'&#9733; Most common',color:C.accent},
      {name:'AdamW',adaptive:'Yes',momentum:'Yes (decoupled)',rec:'LLMs, transformers',def:'&#9733; LLM standard',color:C.purple},
      {name:'AdaGrad',adaptive:'Yes (monotone)',momentum:'No',rec:'Sparse NLP features',def:'Legacy',color:C.orange},
    ].map(function(r,i){
      var hl=r.name.startsWith('Adam');
      return '<tr style="background:'+(hl?hex(r.color,.06):'transparent')+'">'
        +'<td style="color:'+r.color+';font-weight:700;border-radius:6px 0 0 6px;">'+r.name+'</td>'
        +'<td style="color:'+C.muted+';">'+r.adaptive+'</td>'
        +'<td style="color:'+C.muted+';">'+r.momentum+'</td>'
        +'<td style="color:'+C.muted+';">'+r.rec+'</td>'
        +'<td style="color:'+r.color+';font-weight:700;border-radius:0 6px 6px 0;">'+r.def+'</td>'
        +'</tr>';
    }).join('')
    +'</tbody></table></div>'
  );

  // LR strategies
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Learning Rate Strategies')
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +[
      {name:"Fixed LR",color:C.muted,desc:"Set once and never changed. Simple but suboptimal — you can't be both exploratory early and precise late."},
      {name:"Step Decay",color:C.blue,desc:"Reduce LR by a factor every N epochs (e.g. &times;0.5 every 10 epochs). Simple and widely used in CV."},
      {name:"Cosine Annealing",color:C.accent,desc:"LR follows a cosine curve from high to near-zero. Smooth decay — standard for image classification."},
      {name:"Warmup + Decay",color:C.purple,desc:"Start very small, ramp up over first few thousand steps, then decay. Standard for transformer training — prevents early instability."},
    ].map(function(s){
      return '<div style="flex:1;min-width:170px;padding:10px 14px;border-radius:8px;background:'+hex(s.color,.06)+';border:1px solid '+hex(s.color,.2)+'">'
        +div('font-size:10px;font-weight:700;color:'+s.color+';margin-bottom:6px;',s.name)
        +div('font-size:8px;color:'+C.muted+';line-height:1.7;',s.desc)
        +'</div>';
    }).join('')
    +'</div>'
  );

  out+=insight('&#9881;','The Golden Rule of Learning Rates',
    'Too large: loss explodes or oscillates — the model overshoots the minimum. '
    +'Too small: training converges glacially or stalls in a plateau. '
    +'<strong style="color:'+C.accent+'">Adam defaults (lr=1e-3, &beta;&#8321;=0.9, &beta;&#8322;=0.999) work surprisingly well</strong> as a starting point for most problems. '
    +'For LLMs, use <strong style="color:'+C.purple+'">AdamW + warmup + cosine decay</strong> — this is the standard recipe.');
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 5 — MODEL LANDSCAPE
═══════════════════════════════════════════════════════ */
function renderModels(){
  var types=[{label:"Classification",color:C.accent},{label:"Regression",color:C.blue}];
  var tSel=S.mlType;

  var families=[
    {name:"Linear Models",color:C.muted,icon:"&#8725;",
     classEx:"Logistic Regression, LDA",regEx:"Linear/Ridge/Lasso Regression",
     strengths:"Interpretable, fast, works with small data, baseline for any problem",
     weaknesses:"Assumes linear relationship — fails on complex non-linear data",
     whenUse:"Always start here. If it works, you're done. Explainability required (medicine, finance)."},
    {name:"Tree-Based",color:C.yellow,icon:"&#127795;",
     classEx:"Decision Tree, Random Forest, XGBoost",regEx:"Decision Tree Regression, Gradient Boosting",
     strengths:"Handles non-linear patterns, robust to outliers, interpretable (single tree), no feature scaling needed",
     weaknesses:"Single trees overfit easily. Deep trees are black boxes.",
     whenUse:"Tabular/structured data. XGBoost often beats neural networks on spreadsheet data."},
    {name:"Support Vector",color:C.cyan,icon:"&#8741;",
     classEx:"SVC (Linear, RBF, Poly kernels)",regEx:"SVR",
     strengths:"Excellent on small datasets, effective in high dimensions (text), kernel trick for non-linear",
     weaknesses:"Slow on large datasets (O(n&sup2;) or worse), hard to tune",
     whenUse:"High-dimensional small datasets. Text classification. When you have < 10K examples."},
    {name:"Ensemble",color:C.orange,icon:"&#128101;",
     classEx:"Random Forest, XGBoost, AdaBoost",regEx:"Gradient Boosting, Stacking",
     strengths:"Best predictive performance on tabular data, robust to noise, handles mixed feature types",
     weaknesses:"Less interpretable than single models, slower to train",
     whenUse:"Competition-winning on tabular data. XGBoost/LightGBM are the go-to for structured business data."},
    {name:"Neural Networks",color:C.accent,icon:"&#129504;",
     classEx:"MLP, CNN (images), Transformer (text)",regEx:"MLP Regression, Encoder models",
     strengths:"Scales with data, handles unstructured data (images/text/audio), learns features automatically",
     weaknesses:"Needs huge data, slow to train, black box, requires GPU",
     whenUse:"Images, text, audio, video. When you have > 100K examples. When tabular methods have been tried first."},
    {name:"KNN / Lazy",color:C.pink,icon:"&#128269;",
     classEx:"K-Nearest Neighbors (classification)",regEx:"KNN Regression",
     strengths:"No training phase, simple to understand, non-parametric",
     weaknesses:"Slow at inference (searches all training data), memory-heavy, poor on high dimensions",
     whenUse:"Small datasets, baselines, recommendation prototypes. Rarely in production."},
  ];

  var mSel=S.mlModel;
  var mv=families[mSel];

  var out=sectionTitle("The Model Landscape","From linear regression to deep learning — when and why to use each");

  // Model selector
  out+='<div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center;margin-bottom:20px;">';
  families.forEach(function(f,i){ out+=btnSel(i,mSel,f.color,f.icon+' '+f.name,'mlModel'); });
  out+='</div>';

  // Detail card
  out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+mv.color+';">';
  out+='<div style="display:flex;gap:20px;flex-wrap:wrap;">';
  out+='<div style="flex:2;min-width:260px;">';
  out+=div('font-size:16px;font-weight:800;color:'+mv.color+';margin-bottom:10px;',mv.icon+' '+mv.name);
  out+='<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;">';
  out+='<div style="padding:8px 12px;border-radius:6px;background:'+hex(C.accent,.08)+';border:1px solid '+hex(C.accent,.2)+'">'
    +div('font-size:8px;color:'+C.dim+';margin-bottom:3px;','CLASSIFICATION')
    +div('font-size:9px;color:'+C.accent+';font-family:inherit;',mv.classEx)
    +'</div>';
  out+='<div style="padding:8px 12px;border-radius:6px;background:'+hex(C.blue,.08)+';border:1px solid '+hex(C.blue,.2)+'">'
    +div('font-size:8px;color:'+C.dim+';margin-bottom:3px;','REGRESSION')
    +div('font-size:9px;color:'+C.blue+';font-family:inherit;',mv.regEx)
    +'</div>';
  out+='</div>';
  out+=div('font-size:9px;color:'+C.green+';margin-bottom:4px;','&#10003; Strengths: '+span('color:'+C.muted+';',mv.strengths));
  out+=div('font-size:9px;color:'+C.red+';margin-bottom:4px;margin-top:6px;','&#8722; Weaknesses: '+span('color:'+C.muted+';',mv.weaknesses));
  out+='</div>';
  out+='<div style="min-width:200px;padding:12px 14px;border-radius:8px;background:'+hex(mv.color,.07)+';border:1px solid '+hex(mv.color,.2)+'">';
  out+=div('font-size:9px;font-weight:700;color:'+mv.color+';margin-bottom:6px;','When to Use');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',mv.whenUse);
  out+='</div></div></div>';

  // Mental map
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Complexity vs Interpretability Spectrum')
    +'<div style="position:relative;height:60px;margin:10px 0;">'
    // Arrow
    +'<div style="position:absolute;top:50%;left:0;right:0;height:2px;background:linear-gradient(90deg,'+C.muted+','+C.blue+','+C.accent+');transform:translateY(-50%);"></div>'
    +'<div style="position:absolute;top:50%;right:0;transform:translateY(-50%) translateX(6px);color:'+C.accent+';font-size:14px;">&#9658;</div>'
    // Labels
    +'<div style="display:flex;justify-content:space-between;position:absolute;bottom:0;left:0;right:12px;">'
    +['Linear','KNN/Trees','Ensemble','Deep NN'].map(function(l,i){
      var cols=[C.muted,C.yellow,C.orange,C.accent];
      return '<div style="font-size:8px;color:'+cols[i]+';font-weight:700;">'+l+'</div>';
    }).join('')
    +'</div>'
    +'</div>'
    +'<div style="display:flex;justify-content:space-between;margin-top:4px;">'
    +'<div style="font-size:8px;color:'+C.green+';">&#129504; Most Interpretable | Fast | Small data</div>'
    +'<div style="font-size:8px;color:'+C.accent+';">Most Powerful | Needs GPU + Big data | Black box &#128064;</div>'
    +'</div>'
    +'<div style="margin-top:14px;padding:10px 14px;border-radius:8px;background:'+hex(C.yellow,.06)+';border:1px solid '+hex(C.yellow,.2)+';font-size:9px;color:'+C.muted+';line-height:1.8;">'
    +'<strong style="color:'+C.yellow+'">&#128161; Common misconception:</strong> Neural networks have NOT "won everything." '
    +'For structured/tabular data (spreadsheets, databases), <strong style="color:'+C.orange+'">XGBoost/LightGBM frequently outperform neural networks</strong> '
    +'and are much faster to train. Deep learning wins on unstructured data (images, text, audio) and scales better with massive datasets. '
    +'The right model is always the one that fits your data, not the most sophisticated one.'
    +'</div>'
  );

  out+=insight('&#128200;','Output Type Decision Guide',
    chip('Continuous value',C.blue)+' Linear Reg, SVR, Gradient Boosting &nbsp;|&nbsp; '
    +chip('Binary class',C.accent)+' Logistic Reg, SVM, Random Forest &nbsp;|&nbsp; '
    +chip('Multi-class',C.purple)+' Neural Net, XGBoost, Naive Bayes &nbsp;|&nbsp; '
    +chip('Sequence',C.cyan)+' LSTM, Transformer, CRF &nbsp;|&nbsp; '
    +chip('Ranking',C.orange)+' LambdaMART, RankNet');
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 6 — ALGORITHM ZOO
═══════════════════════════════════════════════════════ */
function renderAlgoZoo(){
  var cats=[
    {name:"Regression",color:C.blue,icon:"&#128200;",algos:[
      {n:"Linear Regression",note:"Ordinary least squares. The baseline for all regression."},
      {n:"Ridge (L2)",note:"Linear + L2 penalty. Shrinks coefficients, prevents overfitting."},
      {n:"Lasso (L1)",note:"Linear + L1 penalty. Drives coefficients to zero — feature selection."},
      {n:"Elastic Net",note:"L1 + L2 combined. Best of both regularization approaches."},
      {n:"Polynomial Regression",note:"Adds polynomial features. Fits curved relationships with linear math."},
      {n:"Bayesian Linear Regression",note:"Places priors on weights. Produces uncertainty estimates."},
      {n:"Quantile Regression",note:"Predicts a quantile (e.g. median) instead of mean. Robust to outliers."},
    ]},
    {name:"Classification",color:C.accent,icon:"&#127921;",algos:[
      {n:"Logistic Regression",note:"Linear + sigmoid. Probabilistic output. Fast, interpretable baseline."},
      {n:"LDA / QDA",note:"Linear/Quadratic Discriminant Analysis. Assumes Gaussian class distributions."},
      {n:"Naive Bayes",note:"Probabilistic. Assumes feature independence. Fast, works well on text."},
      {n:"K-Nearest Neighbors",note:"No training. Classify by majority vote of K nearest examples."},
      {n:"Perceptron",note:"The simplest neural unit. Binary linear classifier. Historical significance."},
    ]},
    {name:"Tree-Based",color:C.yellow,icon:"&#127795;",algos:[
      {n:"Decision Tree (CART)",note:"Binary splits on feature thresholds. Fully interpretable. Overfits easily."},
      {n:"Random Forest",note:"Ensemble of trees on random subsets. Reduces variance. Very robust."},
      {n:"Gradient Boosting",note:"Sequential trees, each corrects prior errors. Very powerful on tabular data."},
      {n:"XGBoost",note:"Regularized gradient boosting with parallel computation. Dominated Kaggle 2015-2020."},
      {n:"LightGBM",note:"Leaf-wise growth + histogram binning. Faster than XGBoost on large datasets."},
      {n:"CatBoost",note:"Native categorical feature handling. Minimal preprocessing needed."},
      {n:"AdaBoost",note:"Weighted ensemble. Boosts misclassified examples. The original boosting algorithm."},
    ]},
    {name:"SVM",color:C.cyan,icon:"&#8741;",algos:[
      {n:"SVC — Linear Kernel",note:"Maximum-margin hyperplane. Best for linearly separable problems."},
      {n:"SVC — RBF Kernel",note:"Radial Basis Function. Handles non-linear boundaries via kernel trick."},
      {n:"SVC — Polynomial Kernel",note:"Polynomial decision boundary. Good for image recognition."},
      {n:"SVR (Regression)",note:"SVM for continuous outputs. Finds tube that contains most points."},
      {n:"Nu-SVC / Nu-SVR",note:"Alternative parameterization using nu to control support vectors."},
    ]},
    {name:"Neural / Deep",color:C.purple,icon:"&#129504;",algos:[
      {n:"MLP (Feedforward)",note:"Multi-Layer Perceptron. Fully connected layers. Universal approximator."},
      {n:"CNN",note:"Convolutional Neural Network. Spatial feature detection. Dominates image tasks."},
      {n:"RNN",note:"Recurrent Neural Network. Processes sequences with memory. Largely replaced by Transformers."},
      {n:"LSTM",note:"Long Short-Term Memory. Gated memory cells. Better gradient flow than vanilla RNN."},
      {n:"Transformer",note:"Attention-based. Handles long-range dependencies. Foundation of all modern LLMs."},
      {n:"Capsule Networks",note:"Groups neurons into capsules preserving spatial relationships. Active research."},
    ]},
    {name:"Ensemble",color:C.orange,icon:"&#128101;",algos:[
      {n:"Bagging",note:"Train N models on bootstrap samples. Average predictions. Reduces variance."},
      {n:"Boosting",note:"Train N models sequentially, each fixing prior errors. Reduces bias."},
      {n:"Stacking",note:"Meta-learner trained on outputs of base models. Often best predictive performance."},
      {n:"Hard Voting",note:"Each model votes; majority wins. Simple, interpretable ensemble."},
      {n:"Soft Voting",note:"Average predicted probabilities. Smoother than hard voting. Usually better."},
    ]},
    {name:"Probabilistic",color:C.pink,icon:"&#127928;",algos:[
      {n:"Gaussian Process",note:"Non-parametric Bayesian model. Provides uncertainty intervals. Expensive at scale."},
      {n:"Bayesian Network",note:"Directed acyclic graph of conditional probabilities. Interpretable causality."},
      {n:"Hidden Markov Model",note:"Sequential probabilistic model. Speech recognition, gene sequence analysis."},
      {n:"Naive Bayes",note:"Strong independence assumption. Works surprisingly well for text classification."},
    ]},
  ];
  var cSel=S.algoCategory;
  var cv=cats[cSel];

  var out=sectionTitle("Algorithm Zoo","Every major supervised learning algorithm — organized by family");

  // Category selector
  out+='<div style="display:flex;gap:4px;flex-wrap:wrap;justify-content:center;margin-bottom:20px;">';
  cats.forEach(function(c,i){ out+=btnSel(i,cSel,c.color,c.icon+' '+c.name,'algoCategory'); });
  out+='</div>';

  // Algorithm cards
  out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+cv.color+';">';
  out+=div('font-size:13px;font-weight:800;color:'+cv.color+';margin-bottom:14px;',cv.icon+' '+cv.name+' Algorithms');
  out+='<div style="display:flex;flex-direction:column;gap:6px;">';
  cv.algos.forEach(function(a){
    out+='<div style="display:flex;gap:12px;align-items:flex-start;padding:10px 14px;border-radius:8px;background:'+hex(cv.color,.05)+';border:1px solid '+hex(cv.color,.15)+'">'
      +'<div style="font-size:10px;font-weight:700;color:'+cv.color+';min-width:200px;flex-shrink:0;">'+a.n+'</div>'
      +div('font-size:9px;color:'+C.muted+';line-height:1.7;',a.note)
      +'</div>';
  });
  out+='</div></div>';

  // Summary table
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Quick Reference — Output Type vs Algorithm Family')
    +'<div style="overflow-x:auto;"><table><thead><tr>'
    +['Output Type','Primary Families','Top Algorithms'].map(function(h){return '<th>'+h+'</th>';}).join('')
    +'</tr></thead><tbody>'
    +[
      {t:'Continuous value',f:'Linear, Tree, SVM',a:'Linear Regression, XGBoost, SVR, Gaussian Process',c:C.blue},
      {t:'Binary class',f:'Linear, Tree, SVM, NN',a:'Logistic Regression, XGBoost, SVC, MLP',c:C.accent},
      {t:'Multi-class',f:'Tree, NN, Probabilistic',a:'Random Forest, Neural Network, Naive Bayes',c:C.purple},
      {t:'Sequence / Structured',f:'Neural, Probabilistic',a:'LSTM, Transformer, CRF, HMM',c:C.cyan},
      {t:'Ranking',f:'Ensemble, NN',a:'LambdaMART, RankNet, RankBoost',c:C.orange},
    ].map(function(r){
      return '<tr>'
        +'<td style="color:'+r.c+';font-weight:700;border-radius:6px 0 0 6px;">'+r.t+'</td>'
        +'<td style="color:'+C.muted+';">'+r.f+'</td>'
        +'<td style="color:'+C.dim+';border-radius:0 6px 6px 0;">'+r.a+'</td>'
        +'</tr>';
    }).join('')
    +'</tbody></table></div>'
  );

  out+=insight('&#128218;','The Three Layers to Keep Separate',
    '<strong style="color:'+C.accent+'">1. Learning Paradigm</strong> — Supervised learning is defined by labeled data + learning an input&rarr;output mapping. This is the category of problem, not the solution. &nbsp;'
    +'<strong style="color:'+C.blue+'">2. Optimization Method</strong> — How the model adjusts parameters. Gradient descent, normal equation, greedy splits, convex optimization, or none (KNN). Gradient descent is one tool among several. &nbsp;'
    +'<strong style="color:'+C.purple+'">3. Model Class</strong> — The type of function learned: linear, tree, ensemble, neural. These are independent choices. '
    +'Conflating these three is the most common confusion in early ML study.');
  return out;
}

/* ═══════════════════════════════════════════════════════
   ROOT RENDER
═══════════════════════════════════════════════════════ */
var TABS=[
  "&#128260; Pipeline",
  "&#127358; Training Loop",
  "&#9878; Bias-Variance",
  "&#9889; Optimizers",
  "&#128270; Model Landscape",
  "&#128218; Algorithm Zoo"
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.blue+','+C.accent+','+C.green+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Supervised Learning Core Idea</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Interactive visual guide — from labeled data to deployed model')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  if(S.tab===0) html+=renderPipeline();
  else if(S.tab===1) html+=renderTraining();
  else if(S.tab===2) html+=renderBiasVariance();
  else if(S.tab===3) html+=renderOptimizers();
  else if(S.tab===4) html+=renderModels();
  else if(S.tab===5) html+=renderAlgoZoo();
  html+='</div>';
  return html;
}

function render(){
  document.getElementById('app').innerHTML=renderApp();
  bindEvents();
}

function bindEvents(){
  document.querySelectorAll('[data-action]').forEach(function(el){
    var action=el.getAttribute('data-action');
    var idx=parseInt(el.getAttribute('data-idx'));
    var tag=el.tagName.toLowerCase();
    if(tag==='button'){
      el.addEventListener('click',function(){
        if(action==='tab') S.tab=idx;
        else if(action==='pipeStep') S.pipeStep=(S.pipeStep===idx?-1:idx);
        else if(action==='trainFwd'){ if(S.trainIter<10) S.trainIter++; }
        else if(action==='trainBack'){ if(S.trainIter>0) S.trainIter--; }
        else if(action==='trainReset') S.trainIter=0;
        else if(action==='optSel') S.optSel=idx;
        else if(action==='mlModel') S.mlModel=idx;
        else if(action==='algoCategory') S.algoCategory=idx;
        render();
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseInt(this.value);
        if(action==='trainIter') S.trainIter=val;
        else if(action==='bvComplexity') S.bvComplexity=val;
        render();
      });
    }
  });
}

render();
</script>
</body>
</html>"""

SL_CORE_HEIGHT = 1800