"""
Self-contained HTML visual for Gradient Boosting.
5 interactive tabs: Intro & Sequential Boosting, Residuals & Pseudo-Residuals,
Learning Rate & Shrinkage, Bias-Variance & n_estimators, Regularisation (depth/subsampling).
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(GB_VISUAL_HTML, height=GB_VISUAL_HEIGHT, scrolling=True)
"""

GB_VISUAL_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
button{cursor:pointer;font-family:inherit;}
input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#4ecdc4;cursor:pointer;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.fade{animation:fadeIn .3s ease both;}
.card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
.tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
.tab-btn{padding:12px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:10px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
.tab-btn.active{border-bottom-color:#4ecdc4;color:#4ecdc4;}
.section-title{text-align:center;margin-bottom:20px;}
.section-title h2{font-size:18px;font-weight:800;margin-bottom:4px;color:#e4e4e7;}
.section-title p{font-size:11px;color:#71717a;}
.insight{max-width:750px;margin:16px auto 0;padding:16px 22px;background:rgba(78,205,196,.06);border-radius:10px;border:1px solid rgba(78,205,196,.2);}
.ins-title{font-size:11px;font-weight:700;color:#4ecdc4;margin-bottom:6px;}
.ins-body{font-size:11px;color:#71717a;line-height:1.8;}
</style>
</head>
<body>
<div id="app" style="max-width:960px;margin:0 auto;padding:24px 16px;"></div>
<script>
/* ─── PALETTE ─── */
var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",
  accent:"#4ecdc4",blue:"#38bdf8",purple:"#a78bfa",
  yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",
  dim:"#3f3f46",red:"#ef4444",green:"#4ade80",
  orange:"#fb923c"};

/* ─── HELPERS ─── */
function hex(c,a){
  var r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
  return 'rgba('+r+','+g+','+b+','+a+')';
}
function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
function card(inner,extra){
  return '<div class="card" style="max-width:750px;margin:0 auto 14px;'+(extra||'')+'">'+inner+'</div>';
}
function sectionTitle(t,s){
  return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';
}
function insight(icon,title,body){
  return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';
}
function btnSel(idx,cur,color,label,action){
  var on=idx===cur;
  return '<button data-action="'+action+'" data-idx="'+idx
    +'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
    +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';
}
function sliderRow(action,val,min,max,step,label,dec){
  var dv=(dec!==undefined)?val.toFixed(dec):val;
  return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
    +'<div style="font-size:10px;color:'+C.muted+';width:80px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:52px;font-weight:700;">'+dv+'</div>'
    +'</div>';
}
function statRow(label,val,color){
  return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<span style="color:'+C.muted+';">'+label+'</span>'
    +'<span style="color:'+color+';font-weight:700;">'+val+'</span></div>';
}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||440)+' '+(h||280)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}

/* ─── SVG PLOT SCAFFOLD ─── */
var VW=440,VH=280,PL=46,PR=16,PT=16,PB=38;
var PW=VW-PL-PR, PH=VH-PT-PB;
function sx(x,xmax){return PL+((x)/(xmax||10))*PW;}
function sy(y,ymax){return PT+PH-((y)/(ymax||10))*PH;}
function plotAxes(xl,yl,xmax,ymax,xticks,yticks){
  var xm=xmax||10, ym=ymax||10;
  var xts=xticks||[0,2,4,6,8,10], yts=yticks||[0,2,4,6,8,10];
  var o='';
  xts.forEach(function(v){
    o+='<line x1="'+sx(v,xm).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xm).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  yts.forEach(function(v){
    o+='<line x1="'+PL+'" y1="'+sy(v,ym).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ym).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  xts.forEach(function(v){
    o+='<text x="'+sx(v,xm).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  yts.forEach(function(v){
    o+='<text x="'+(PL-6)+'" y="'+(sy(v,ym)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xl+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
  return o;
}

/* ─── STATE ─── */
var S={
  tab:0,
  boostStep:0,    /* intro: 0..4 boosting rounds */
  residStep:0,    /* residuals tab: which round 0..3 */
  lr:0.1,         /* learning rate tab: 0.01..1.0 */
  lrNEst:50,      /* learning rate tab: n_estimators for lr comparison */
  nEst:50,        /* bias-variance: n_estimators 1..300 */
  maxDepth:3,     /* regularisation tab: max_depth 1..8 */
  subsample:1.0   /* regularisation tab: subsample 0.5..1.0 */
};

/* ══════════════════════════════════════════════════════════
   SHARED DATA  — 12 regression points  y = sin(x) + noise
══════════════════════════════════════════════════════════ */
var XS=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0];
var YS=[1.6,2.5,3.2,3.8,4.1,3.9,3.4,2.7,2.0,1.4,1.1,1.3];
var YMEAN=YS.reduce(function(a,b){return a+b;})/YS.length; /* ~2.67 */

/* Pre-computed ensemble predictions after each round (lr=0.1, depth=1 stumps) */
/* F0 = mean, then each round adds a weak learner correction */
var ROUND_PREDS=[
  /* round 0: F0 = mean for all */
  [2.67,2.67,2.67,2.67,2.67,2.67,2.67,2.67,2.67,2.67,2.67,2.67],
  /* round 1: after 1 tree */
  [2.60,2.74,2.88,3.10,3.22,3.22,3.10,2.88,2.74,2.60,2.52,2.46],
  /* round 2 */
  [2.42,2.72,3.05,3.45,3.71,3.68,3.44,3.02,2.64,2.34,2.18,2.20],
  /* round 3 */
  [1.92,2.62,3.21,3.72,3.98,3.96,3.70,3.19,2.59,2.14,1.86,1.90],
  /* round 4 */
  [1.65,2.52,3.21,3.78,4.06,4.02,3.74,3.19,2.50,1.97,1.62,1.62]
];

function mse(preds){
  var s=0;
  for(var i=0;i<YS.length;i++) s+=Math.pow(YS[i]-preds[i],2);
  return s/YS.length;
}

/* ══════════════════════════════════════════════════════════
   TAB 0 — INTRO & SEQUENTIAL BOOSTING
══════════════════════════════════════════════════════════ */
var ROUND_COLS=[C.dim,C.blue,C.orange,C.purple,C.green];
var ROUND_LABELS=['F\u2080 (mean)','+ Tree 1','+ Tree 2','+ Tree 3','+ Tree 4'];

function renderIntro(){
  var step=S.boostStep;
  var preds=ROUND_PREDS[step];
  var curMSE=mse(preds);
  var initMSE=mse(ROUND_PREDS[0]);

  /* ─── regression fit plot ─── */
  var sv=plotAxes('x','y',6.5,5,
    [0,1,2,3,4,5,6],[0,1,2,3,4,5]);

  /* true data points */
  XS.forEach(function(x,i){
    sv+='<circle cx="'+sx(x,6.5).toFixed(1)+'" cy="'+sy(YS[i],5).toFixed(1)+'" r="5"'
      +' fill="'+C.accent+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  });

  /* draw ALL previous ensemble lines faintly */
  for(var r=0;r<step;r++){
    var rp=ROUND_PREDS[r];
    var path='';
    XS.forEach(function(x,i){
      path+=(i===0?'M':'L')+sx(x,6.5).toFixed(1)+','+sy(rp[i],5).toFixed(1)+' ';
    });
    sv+='<path d="'+path+'" fill="none" stroke="'+ROUND_COLS[r]+'" stroke-width="1" opacity="0.3"/>';
  }

  /* current prediction line */
  var curPath='';
  XS.forEach(function(x,i){
    curPath+=(i===0?'M':'L')+sx(x,6.5).toFixed(1)+','+sy(preds[i],5).toFixed(1)+' ';
  });
  sv+='<path d="'+curPath+'" fill="none" stroke="'+ROUND_COLS[step]+'" stroke-width="2.5"/>';

  /* residual arrows */
  XS.forEach(function(x,i){
    var py=sy(preds[i],5), ty=sy(YS[i],5);
    if(Math.abs(YS[i]-preds[i])>0.05){
      var arrowCol=YS[i]>preds[i]?C.green:C.red;
      sv+='<line x1="'+sx(x,6.5).toFixed(1)+'" y1="'+py.toFixed(1)+'"'
        +' x2="'+sx(x,6.5).toFixed(1)+'" y2="'+ty.toFixed(1)+'"'
        +' stroke="'+arrowCol+'" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.7"/>';
    }
  });

  /* MSE badge */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="128" height="20" rx="4" fill="#0a0a0f" opacity="0.92"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+15)+'" fill="'+ROUND_COLS[step]+'" font-size="10" font-family="monospace" font-weight="700">MSE: '+curMSE.toFixed(3)+'</text>';

  /* ─── sequential tree diagram ─── */
  var TW=440, TH=88;
  var tv='';
  var treeW=74, treeH=32, treeGap=8;
  var totalW=(treeW+treeGap)*5-treeGap;
  var startX=(TW-totalW)/2;
  for(var t=0;t<5;t++){
    var tx=startX+t*(treeW+treeGap);
    var active=t<=step;
    var col=active?ROUND_COLS[t]:C.dim;
    tv+='<rect x="'+tx.toFixed(1)+'" y="20" width="'+treeW+'" height="'+treeH+'" rx="6"'
      +' fill="'+hex(col,active?0.15:0.04)+'" stroke="'+col+'" stroke-width="'+(t===step?2:1)+'"/>';
    tv+='<text x="'+(tx+treeW/2).toFixed(1)+'" y="33" text-anchor="middle" fill="'+col+'" font-size="9" font-weight="700">'
      +(t===0?'F\u2080':('h\u2080'+'₀₁₂₃₄'[t-1]))+'</text>';
    tv+='<text x="'+(tx+treeW/2).toFixed(1)+'" y="44" text-anchor="middle" fill="'+C.muted+'" font-size="7.5">'
      +(t===0?'mean':(active?'residuals ':'\u2026'))+'</text>';
    if(active&&t>0){
      tv+='<text x="'+(tx+treeW/2).toFixed(1)+'" y="55" text-anchor="middle" fill="'+col+'" font-size="7">'
        +'MSE\u2193'+((mse(ROUND_PREDS[t-1])-mse(ROUND_PREDS[t]))*100).toFixed(1)+'%</text>';
    }
    /* arrow between trees */
    if(t<4){
      var ax=tx+treeW+2, ay=20+treeH/2;
      tv+='<text x="'+(ax+3).toFixed(1)+'" y="'+(ay+3)+'" fill="'+C.muted+'" font-size="9">+</text>';
    }
  }
  /* additive formula */
  tv+='<text x="'+TW/2+'" y="76" text-anchor="middle" fill="'+C.muted+'" font-size="8.5" font-family="monospace">'
    +'F\u2098(x) = F\u2080(x) + \u03b7\u00b7h\u2081(x) + \u03b7\u00b7h\u2082(x) + ... + \u03b7\u00b7h\u2098(x)</text>';

  var out=sectionTitle('Sequential Boosting','Each tree corrects the errors of the ensemble so far — building a strong learner from weak ones');

  /* step buttons */
  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  ROUND_LABELS.forEach(function(lbl,i){
    out+=btnSel(i,step,ROUND_COLS[i],lbl,'boostStep');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:6px;','Ensemble Fit After Round '+step)
    +svgBox(sv,VW,VH)
    +'<div style="margin-top:6px;">'+svgBox(tv,TW,TH)+'</div>'
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;">'
    +[{c:C.accent,l:'Training data'},{c:C.green,l:'Under-predicted (residual > 0)'},{c:C.red,l:'Over-predicted'}].map(function(it){
      return '<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
        +'<div style="width:10px;height:10px;border-radius:50%;background:'+it.c+'"></div>'+it.l+'</div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','ROUND '+step+' METRICS');
  out+=statRow('Trees added so far',step,C.accent);
  out+=statRow('Current MSE',curMSE.toFixed(4),C.red);
  out+=statRow('Initial MSE (F\u2080)',initMSE.toFixed(4),C.yellow);
  var red=step>0?((initMSE-curMSE)/initMSE*100).toFixed(1):0;
  out+=statRow('MSE reduction',red+'%',step>0?C.green:C.muted);
  out+=statRow('Residuals remain',step===4?'Small':'Large',step===4?C.green:C.orange);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','GB vs RANDOM FOREST');
  [
    {lbl:'Training order', gb:'Sequential (must wait)',       rf:'Parallel (independent)', gc:C.red,   rc:C.green},
    {lbl:'Each tree trains on', gb:'Residuals of ensemble',   rf:'Bootstrap sample',       gc:C.yellow,rc:C.blue},
    {lbl:'Tree depth',     gb:'Shallow (depth 3\u20136)',     rf:'Full (no limit)',         gc:C.accent,rc:C.muted},
    {lbl:'Reduces',        gb:'Bias (sequential correction)', rf:'Variance (averaging)',    gc:C.purple,rc:C.purple},
    {lbl:'Overfit risk',   gb:'High (needs regularisation)',  rf:'Low (averaging is safe)', gc:C.red,   rc:C.green},
  ].forEach(function(r){
    out+='<div style="padding:4px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="font-size:8.5px;color:'+C.dim+';margin-bottom:2px;">'+r.lbl+'</div>'
      +'<div style="display:flex;gap:8px;">'
      +'<div style="flex:1;font-size:8.5px;color:'+r.gc+';">&#128200; GB: '+r.gb+'</div>'
      +'<div style="flex:1;font-size:8.5px;color:'+r.rc+';">&#127795; RF: '+r.rf+'</div>'
      +'</div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+='<div style="display:flex;gap:10px;justify-content:center;margin-bottom:12px;">';
  out+='<button data-action="boostPrev" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(step>0?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
    +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Prev round</button>';
  out+='<button data-action="boostNext" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(step<4?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step<4?C.accent:C.border)+';'
    +'color:'+(step<4?C.accent:C.dim)+';cursor:'+(step<4?'pointer':'default')+';">Next round \u2192</button>';
  out+='</div>';

  out+=insight('&#128200;','Boosting = Gradient Descent in Function Space',
    'Each tree is a <span style="color:'+C.yellow+';font-weight:700;">step in gradient descent</span> — '
    +'not on model parameters, but on the <em>prediction function itself</em>. '
    +'The pseudo-residuals are the negative gradient of the loss w.r.t. the current predictions. '
    +'By fitting each new tree to these residuals we move the ensemble in the direction that '
    +'<span style="color:'+C.green+';font-weight:700;">steepest reduces the loss</span>.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 1 — RESIDUALS & PSEUDO-RESIDUALS
══════════════════════════════════════════════════════════ */
function renderResiduals(){
  var step=S.residStep;
  var preds=ROUND_PREDS[step];
  var nextPreds=ROUND_PREDS[Math.min(step+1,4)];

  /* residuals = y - F(x) */
  var resids=YS.map(function(y,i){return y-preds[i];});
  var absMax=Math.max.apply(null,resids.map(Math.abs));

  /* ─── left plot: current residuals ─── */
  var rMin=-2, rMax=2;
  var sv=plotAxes('x','Residual r = y \u2212 F(x)',6.5,rMax,
    [0,1,2,3,4,5,6],[-2,-1,0,1,2]);

  /* zero line */
  sv+='<line x1="'+PL+'" y1="'+sy(0,rMax).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(0,rMax).toFixed(1)+'"'
    +' stroke="'+C.accent+'" stroke-width="1" stroke-dasharray="5,3" opacity="0.5"/>';

  /* residual bars */
  XS.forEach(function(x,i){
    var r=resids[i];
    var barTop=sy(Math.max(r,0),rMax), barBot=sy(Math.min(r,0),rMax);
    var col=r>0?C.green:C.red;
    sv+='<rect x="'+(sx(x,6.5)-6).toFixed(1)+'" y="'+barTop.toFixed(1)+'"'
      +' width="12" height="'+(barBot-barTop).toFixed(1)+'" fill="'+hex(col,0.7)+'" rx="2"/>';
    /* next tree's prediction for this residual */
    var nextR=YS[i]-nextPreds[i];
    var newRCol=Math.abs(nextR)<Math.abs(r)?C.green:C.red;
    sv+='<circle cx="'+sx(x,6.5).toFixed(1)+'" cy="'+sy(YS[i]-nextPreds[i],rMax).toFixed(1)+'" r="3"'
      +' fill="'+newRCol+'" stroke="#0a0a0f" stroke-width="1" opacity="0.8"/>';
  });

  /* stump line — the weak learner fitted to residuals */
  /* approximate: two-region constant prediction */
  var leftMean=resids.slice(0,6).reduce(function(a,b){return a+b;})/6;
  var rightMean=resids.slice(6).reduce(function(a,b){return a+b;})/6;
  sv+='<line x1="'+sx(0,6.5).toFixed(1)+'" y1="'+sy(leftMean,rMax).toFixed(1)+'"'
    +' x2="'+sx(3.25,6.5).toFixed(1)+'" y2="'+sy(leftMean,rMax).toFixed(1)+'"'
    +' stroke="'+C.yellow+'" stroke-width="2.5"/>';
  sv+='<line x1="'+sx(3.25,6.5).toFixed(1)+'" y1="'+sy(rightMean,rMax).toFixed(1)+'"'
    +' x2="'+sx(6.5,6.5).toFixed(1)+'" y2="'+sy(rightMean,rMax).toFixed(1)+'"'
    +' stroke="'+C.yellow+'" stroke-width="2.5"/>';
  sv+='<line x1="'+sx(3.25,6.5).toFixed(1)+'" y1="'+PT+'"'
    +' x2="'+sx(3.25,6.5).toFixed(1)+'" y2="'+(PT+PH)+'"'
    +' stroke="'+C.yellow+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>';

  /* legend */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="165" height="38" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  [[C.green,'Positive residual (underfit)'],[C.red,'Negative residual (overfit)'],
   [C.yellow,'New tree fit to residuals']].forEach(function(item,li){
    sv+='<rect x="'+(PL+8)+'" y="'+(PT+8+li*11)+'" width="8" height="8" rx="2" fill="'+item[0]+'"/>';
    sv+='<text x="'+(PL+20)+'" y="'+(PT+15+li*11)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">'+item[1]+'</text>';
  });

  /* ─── right panel ─── */
  var rmseCur=Math.sqrt(mse(preds)), rmseNext=Math.sqrt(mse(nextPreds));

  var out=sectionTitle('Residuals & Pseudo-Residuals','The new tree fits the errors of the current ensemble — under MSE loss these equal the raw residuals');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  ['Round 0','Round 1','Round 2','Round 3'].forEach(function(lbl,i){
    out+=btnSel(i,step,ROUND_COLS[i+1],lbl,'residStep');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Residuals After Round '+step+' (yellow = next weak learner)')
    +svgBox(sv)
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','ROUND '+step+' \u2192 ROUND '+(step+1));
  out+=statRow('RMSE before',rmseCur.toFixed(4),C.red);
  out+=statRow('RMSE after (est.)',rmseNext.toFixed(4),C.green);
  out+=statRow('Reduction',((rmseCur-rmseNext)/rmseCur*100).toFixed(1)+'%',C.accent);
  var maxResid=Math.max.apply(null,resids.map(Math.abs));
  out+=statRow('Max |residual|',maxResid.toFixed(3),C.orange);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PSEUDO-RESIDUALS BY LOSS');
  [
    {loss:'MSE (regression)',   formula:'r\u1d35 = y\u1d35 \u2212 F(x\u1d35)',          col:C.blue,  note:'= raw residuals'},
    {loss:'MAE (regression)',   formula:'r\u1d35 = sign(y\u1d35 \u2212 F(x\u1d35))',    col:C.green, note:'\u00b11 only'},
    {loss:'Log-loss (classif.)',formula:'r\u1d35 = y\u1d35 \u2212 \u03c3(F(x\u1d35))', col:C.purple,note:'= y \u2212 p\u0302'},
    {loss:'Huber',              formula:'r\u1d35 = clip(y\u2212F, \u2212\u03b4, \u03b4)',col:C.yellow,note:'outlier robust'},
  ].forEach(function(r){
    out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.col+';">'
      +'<div style="font-size:8.5px;font-weight:700;color:'+r.col+';">'+r.loss+'</div>'
      +'<div style="font-size:8.5px;font-family:monospace;color:'+C.text+';margin-top:2px;">'+r.formula+'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';">'+r.note+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','THE GB UPDATE RULE');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:2;font-family:monospace;',
    '<span style="color:'+C.text+';">F\u2098(x)</span> = F\u2098\u208b\u2081(x) + <span style="color:'+C.yellow+'">\u03b7</span>\u00b7<span style="color:'+C.orange+'">h\u2098(x)</span><br>'
    +'<span style="color:'+C.muted+';font-size:8px;">\u03b7 = learning rate &nbsp; h\u2098 = new tree fit to r\u1d35</span><br>'
    +'<span style="color:'+C.accent+'">r\u1d35</span> = \u2212\u2202L/\u2202F(x\u1d35) &nbsp; <span style="color:'+C.muted+';font-size:8px;">(negative gradient)</span>'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#129518;','Why Pseudo-Residuals = Negative Gradient',
    'For MSE loss L = \u00bd(y\u2212F)\u00b2, the derivative w.r.t. F is <span style="color:'+C.red+';font-family:monospace;">\u2202L/\u2202F = \u2212(y\u2212F)</span>. '
    +'The <em>negative</em> gradient is <span style="color:'+C.green+';font-family:monospace;">y \u2212 F</span> — which is exactly the residual. '
    +'For other losses (log-loss, Huber) the pseudo-residuals differ from raw residuals, '
    +'but the principle is the same: <span style="color:'+C.accent+';font-weight:700;">fit the next tree to the negative gradient</span> of the loss.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 2 — LEARNING RATE & SHRINKAGE
══════════════════════════════════════════════════════════ */
function gbTestErr(n,lr){
  /* simulated: lower lr needs more trees; convergence level same */
  var k=lr*0.9+0.05;
  return 0.04+0.55*Math.exp(-k*n*0.04);
}
function gbTrainErr(n,lr){
  var k=lr*1.1+0.08;
  return 0.01+0.05*Math.exp(-k*n*0.03);
}

function renderLearningRate(){
  var lr=S.lr;
  var nE=S.lrNEst;
  var trE=gbTrainErr(nE,lr), teE=gbTestErr(nE,lr);
  var lrLow=0.01, lrHigh=1.0;

  /* ─── 3-curve plot: lr=0.01, current, lr=1.0 ─── */
  var sv=plotAxes('n_estimators','Test Error',300,0.6,
    [0,50,100,150,200,250,300],[0,0.1,0.2,0.3,0.4,0.5,0.6]);

  var lrPairs=[
    {lr:0.01,col:C.blue,  lbl:'\u03b7=0.01 (too small)'},
    {lr:lr,   col:C.accent,lbl:'\u03b7='+lr.toFixed(2)+' (current)'},
    {lr:1.0,  col:C.red,  lbl:'\u03b7=1.0 (too large)'}
  ];
  lrPairs.forEach(function(pair){
    var path='';
    for(var n=1;n<=300;n+=3){
      var ex=sx(n,300), ey=sy(gbTestErr(n,pair.lr),0.6);
      path+=(n===1?'M':'L')+ex.toFixed(1)+','+ey.toFixed(1)+' ';
    }
    sv+='<path d="'+path+'" fill="none" stroke="'+pair.col+'" stroke-width="'+(pair.lr===lr?2.5:1.5)+'"'
      +' stroke-dasharray="'+(pair.lr===lr?'':'6,3')+'" opacity="'+(pair.lr===lr?1:0.5)+'"/>';
  });

  /* current n_est cursor */
  var cvx=sx(nE,300);
  sv+='<line x1="'+cvx.toFixed(1)+'" y1="'+PT+'" x2="'+cvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.yellow+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+sy(teE,0.6).toFixed(1)+'" r="4" fill="'+C.accent+'" stroke="#0a0a0f" stroke-width="1.5"/>';

  /* legend */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="160" height="42" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  lrPairs.forEach(function(p,li){
    sv+='<line x1="'+(PL+8)+'" y1="'+(PT+10+li*13)+'" x2="'+(PL+26)+'" y2="'+(PT+10+li*13)+'" stroke="'+p.col+'" stroke-width="2" '+(p.lr!==lr?'stroke-dasharray="5,2"':'')+'/>';
    sv+='<text x="'+(PL+30)+'" y="'+(PT+14+li*13)+'" fill="'+p.col+'" font-size="8" font-family="monospace">'+p.lbl+'</text>';
  });

  var overfit=teE>0.12?'High':teE>0.07?'Moderate':'Low';
  var ovCol=teE>0.12?C.red:teE>0.07?C.yellow:C.green;
  var converge=gbTestErr(300,lr)<gbTestErr(50,lr)*0.7;

  var out=sectionTitle('Learning Rate (\u03b7) & Shrinkage','Smaller \u03b7 = smaller steps = more trees needed, but better generalisation via shrinkage');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Test Error vs n_estimators (varying \u03b7)')
    +svgBox(sv)
    +sliderRow('lr',lr,0.01,1.0,0.01,'learn rate',2)
    +sliderRow('lrNEst',nE,1,300,1,'n_estimators',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.blue+';">\u2190 \u03b7=0.01 (slow but stable)</span>'
    +'<span style="color:'+C.red+';">\u03b7=1.0 (fast, overfit) \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT (\u03b7='+lr.toFixed(2)+', n='+nE+')');
  out+=statRow('Train error',(trE*100).toFixed(1)+'%',C.green);
  out+=statRow('Test error',(teE*100).toFixed(1)+'%',C.red);
  out+=statRow('Overfit risk',overfit,ovCol);
  out+=statRow('Converged?',converge?'Yes':'Not yet',converge?C.green:C.yellow);
  var optN=Math.round(Math.log(0.55/0.04)/(lr*0.9+0.05)*25);
  out+=statRow('Est. optimal n',optN,C.accent);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','\u03b7 TRADEOFF');
  [
    {lr:'Small (0.01\u20130.05)',pro:'Strong regularisation, robust',con:'Needs many trees (slow)',c:C.blue},
    {lr:'Medium (0.05\u20130.2)',pro:'Good balance',               con:'Standard starting point',c:C.accent},
    {lr:'Large (>0.5)',         pro:'Fast convergence',           con:'Likely overfit on small data',c:C.red},
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">\u03b7 '+r.lr+'</div>'
      +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+r.pro+'</div>'
      +'<div style="font-size:8px;color:'+C.red+';">\u2718 '+r.con+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','SHRINKAGE FORMULA');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:2;font-family:monospace;',
    'F\u2098(x) = F\u2098\u208b\u2081(x) + <span style="color:'+C.yellow+';">\u03b7</span>\u00b7h\u2098(x)<br>'
    +'<span style="color:'+C.muted+';font-size:8px;">As \u03b7\u21920: each step shrinks to 0.</span><br>'
    +'<span style="color:'+C.muted+';font-size:8px;">Shrinkage \u21d4 L2 regularisation</span><br>'
    +'<span style="color:'+C.accent+';font-size:8px;">Rule: halve \u03b7 \u21d2 double n_estimators</span>'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9878;&#65039;','The \u03b7 \u00d7 n_estimators Tradeoff',
    'There is a near-perfect tradeoff: <span style="color:'+C.yellow+';font-weight:700;">halving \u03b7 and doubling n_estimators</span> gives approximately the same test error. '
    +'Small \u03b7 regularises more aggressively (shrinkage effect) — this is why low \u03b7 often generalises better given enough trees. '
    +'In practice: set <span style="color:'+C.accent+';font-family:monospace;">\u03b7=0.05\u20130.1</span> and tune n_estimators with '
    +'<span style="color:'+C.blue+';font-family:monospace;">early_stopping_rounds</span>.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 3 — BIAS-VARIANCE & OVERFITTING
══════════════════════════════════════════════════════════ */
function gbErrN(n){
  /* GB CAN overfit with large n (unlike RF) */
  var base=0.04+0.50*Math.exp(-0.03*n);
  var overfit=n>150?0.0008*(n-150):0;
  return base+overfit;
}
function gbTrainN(n){return 0.005+0.03*Math.exp(-0.05*n);}

function renderBiasVariance(){
  var nE=S.nEst;
  var trE=gbTrainN(nE), teE=gbErrN(nE);
  var optN=130; /* approx min test error */

  /* ─── error curve ─── */
  var sv=plotAxes('n_estimators','Error',300,0.55,
    [0,50,100,150,200,250,300],[0,0.1,0.2,0.3,0.4,0.5]);

  /* optimum band */
  sv+='<rect x="'+sx(100,300).toFixed(1)+'" y="'+PT+'" width="'+(sx(165,300)-sx(100,300)).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.green,0.04)+'"/>';
  sv+='<text x="'+sx(132,300).toFixed(1)+'" y="'+(PT+13)+'" text-anchor="middle" fill="'+C.green+'" font-size="8">sweet spot</text>';

  var trnPath='', tstPath='';
  for(var n=1;n<=300;n+=2){
    var px=sx(n,300);
    trnPath+=(n===1?'M':'L')+px.toFixed(1)+','+sy(gbTrainN(n),0.55).toFixed(1)+' ';
    tstPath+=(n===1?'M':'L')+px.toFixed(1)+','+sy(gbErrN(n),0.55).toFixed(1)+' ';
  }
  sv+='<path d="'+trnPath+'" fill="none" stroke="'+C.green+'" stroke-width="2"/>';
  sv+='<path d="'+tstPath+'" fill="none" stroke="'+C.red+'" stroke-width="2"/>';

  /* cursor */
  var cvx=sx(nE,300);
  sv+='<line x1="'+cvx.toFixed(1)+'" y1="'+PT+'" x2="'+cvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+sy(trE,0.55).toFixed(1)+'" r="4" fill="'+C.green+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+sy(teE,0.55).toFixed(1)+'" r="4" fill="'+C.red+'" stroke="#0a0a0f" stroke-width="1.5"/>';

  /* legend */
  sv+='<rect x="'+(PL+PW-110)+'" y="'+(PT+3)+'" width="108" height="30" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  [[C.green,'Train error'],[C.red,'Test error']].forEach(function(item,li){
    sv+='<line x1="'+(PL+PW-106)+'" y1="'+(PT+10+li*13)+'" x2="'+(PL+PW-88)+'" y2="'+(PT+10+li*13)+'" stroke="'+item[0]+'" stroke-width="2"/>';
    sv+='<text x="'+(PL+PW-84)+'" y="'+(PT+14+li*13)+'" fill="'+item[0]+'" font-size="8" font-family="monospace">'+item[1]+'</text>';
  });

  var gap=teE-trE;
  var regime=nE<40?'High bias':nE<optN?'Converging':nE<200?'Near-optimal':'Overfitting';
  var regimeCol=nE<40?C.orange:nE<optN?C.yellow:nE<200?C.green:C.red;

  var out=sectionTitle('Bias-Variance & Overfitting','Unlike Random Forests, GB CAN overfit with too many estimators — use early stopping');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Train vs Test Error by n_estimators')
    +svgBox(sv)
    +sliderRow('nEst',nE,1,300,1,'n_estimators',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.orange+';">\u2190 high bias (few trees)</span>'
    +'<span style="color:'+C.red+';">overfit (too many) \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','AT n_estimators='+nE);
  out+=statRow('Train error',(trE*100).toFixed(1)+'%',C.green);
  out+=statRow('Test error',(teE*100).toFixed(1)+'%',C.red);
  out+=statRow('Train-test gap',(gap*100).toFixed(1)+'pp',gap>0.05?C.red:C.yellow);
  out+=statRow('Regime',regime,regimeCol);
  out+=statRow('Est. optimal n',optN,C.accent);
  out+=statRow('Overfit risk',nE>optN?'Yes \u26a0':'No',nE>optN?C.red:C.green);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHY GB CAN OVERFIT');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'RF averages independent trees \u2014 more trees only reduce variance and cannot hurt. '
    +'GB fits trees <em>sequentially to residuals</em> of the current ensemble. '
    +'Eventually it begins fitting <span style="color:'+C.red+';font-weight:700;">noise</span> in the residuals. '
    +'<br><br>Solution: <span style="color:'+C.green+';font-family:monospace;">early_stopping_rounds</span> + a validation set.'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','EARLY STOPPING');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:8.5px;color:'+C.muted+';line-height:1.8;',
    '1. Reserve ~20% as validation set<br>'
    +'2. Monitor val loss each round<br>'
    +'3. Stop if no improvement for<br>'
    +'&nbsp;&nbsp;&nbsp;<span style="color:'+C.accent+';font-family:monospace;">early_stopping_rounds</span> rounds<br>'
    +'4. Revert to best round<br><br>'
    +'XGBoost, LightGBM, CatBoost all<br>'
    +'support this natively.'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9203;','Use Early Stopping, Not a Fixed n',
    'Never fix <span style="color:'+C.yellow+';font-family:monospace;">n_estimators</span> without early stopping. '
    +'Set it large (e.g., 5000) and let the algorithm stop when val loss stops improving. '
    +'The result: you get the <span style="color:'+C.green+';font-weight:700;">optimal n automatically</span>, '
    +'robust to dataset size and learning rate. '
    +'In XGBoost: <span style="color:'+C.accent+';font-family:monospace;">early_stopping_rounds=50</span>. '
    +'In sklearn GBM: <span style="color:'+C.accent+';font-family:monospace;">n_iter_no_change=10</span>.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 4 — REGULARISATION (depth + subsampling)
══════════════════════════════════════════════════════════ */
function gbErrDepth(d,ss){
  var base=0.22*Math.exp(-0.45*d)+0.04;
  var overfit=d>4?0.02*(d-4):0;
  var ssBonus=(1-ss)*0.06;
  return base+overfit+ssBonus;
}

function renderRegularisation(){
  var md=S.maxDepth;
  var ss=S.subsample;

  /* depth-error plot */
  var sv=plotAxes('max_depth','Test Error',8,0.4,
    [1,2,3,4,5,6,7,8],[0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]);

  /* subsample curves */
  var ssVals=[1.0,0.8,0.6];
  var ssCols=[C.red,C.orange,C.green];
  var ssLabels=['sub=1.0','sub=0.8','sub=0.6'];
  ssVals.forEach(function(ssv,si){
    var path='';
    for(var d=1;d<=8;d++){
      path+=(d===1?'M':'L')+sx(d,8).toFixed(1)+','+sy(gbErrDepth(d,ssv),0.4).toFixed(1)+' ';
    }
    sv+='<path d="'+path+'" fill="none" stroke="'+ssCols[si]+'" stroke-width="'+(Math.abs(ssv-ss)<0.15?2.5:1.2)+'"'
      +' stroke-dasharray="'+(Math.abs(ssv-ss)<0.15?'':'5,3')+'" opacity="'+(Math.abs(ssv-ss)<0.15?1:0.5)+'"/>';
  });

  /* current marker */
  sv+='<circle cx="'+sx(md,8).toFixed(1)+'" cy="'+sy(gbErrDepth(md,ss),0.4).toFixed(1)+'" r="5" fill="'+C.accent+'" stroke="#0a0a0f" stroke-width="2"/>';

  /* legend */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="100" height="44" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  ssVals.forEach(function(ssv,si){
    sv+='<line x1="'+(PL+8)+'" y1="'+(PT+10+si*13)+'" x2="'+(PL+26)+'" y2="'+(PT+10+si*13)+'" stroke="'+ssCols[si]+'" stroke-width="2" '+(si>0?'stroke-dasharray="5,2"':'')+'/>';
    sv+='<text x="'+(PL+30)+'" y="'+(PT+14+si*13)+'" fill="'+ssCols[si]+'" font-size="8">'+ssLabels[si]+'</text>';
  });

  var curErr=gbErrDepth(md,ss);
  var errColor=curErr<0.08?C.green:curErr<0.15?C.yellow:C.red;
  var regime=md<=2?'Underfitting (too shallow)':md<=4?'Good range':md<=6?'Watch overfit':'Likely overfit';
  var regimeCol=md<=2?C.orange:md<=4?C.green:md<=6?C.yellow:C.red;

  var out=sectionTitle('Regularisation: Depth & Subsampling','Shallow trees prevent overfit; stochastic subsampling adds noise to decorrelate, like RF');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Test Error vs max_depth (colored by subsample)')
    +svgBox(sv)
    +sliderRow('maxDepth',md,1,8,1,'max_depth',0)
    +sliderRow('subsample',ss,0.5,1.0,0.05,'subsample',2)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.orange+';">\u2190 too shallow (high bias)</span>'
    +'<span style="color:'+C.red+';">too deep (overfit) \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT SETTINGS');
  out+=statRow('max_depth',md,C.accent);
  out+=statRow('subsample',ss.toFixed(2),C.blue);
  out+=statRow('Est. test error',(curErr*100).toFixed(1)+'%',errColor);
  out+=statRow('Depth regime',regime,regimeCol);
  out+=statRow('Stochastic?',ss<1.0?'Yes (SGBoost)':'No (full data)',ss<1.0?C.green:C.muted);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','ALL REGULARISATION KNOBS');
  [
    {p:'max_depth',     desc:'tree depth (most important)',    eg:'3\u20136',      c:C.accent},
    {p:'learning_rate', desc:'shrinkage per step',             eg:'0.05\u20130.1', c:C.yellow},
    {p:'subsample',     desc:'fraction of data per tree',      eg:'0.7\u20130.9',  c:C.blue},
    {p:'colsample_bytree',desc:'fraction of features per tree',eg:'0.7\u20130.9',  c:C.purple},
    {p:'min_samples_leaf',desc:'min samples per leaf',         eg:'5\u201320',     c:C.orange},
    {p:'reg_lambda',    desc:'L2 leaf-weight penalty (XGB)',   eg:'0\u20131',      c:C.green},
    {p:'alpha',         desc:'L1 leaf-weight penalty (XGB)',   eg:'0\u20131',      c:C.red},
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:2px 0;border-radius:4px;border-left:2px solid '+r.c+';">'
      +'<div style="display:flex;justify-content:space-between;font-size:9px;">'
      +'<span style="color:'+r.c+';font-family:monospace;">'+r.p+'</span>'
      +'<span style="color:'+C.muted+';">'+r.eg+'</span></div>'
      +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+r.desc+'</div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9881;&#65039;','max_depth is the Most Powerful Knob',
    '<span style="color:'+C.accent+';font-weight:700;">max_depth=3 to 6</span> is almost always the right range for gradient boosting. '
    +'Shallow trees are <em>weak learners</em> — they have high bias but are easy to correct in subsequent rounds. '
    +'Deep trees capture interactions but memorise noise. '
    +'<span style="color:'+C.yellow+';font-family:monospace;">subsample &lt; 1.0</span> turns GB into <em>Stochastic Gradient Boosting</em> — '
    +'each tree sees a random fraction of data, adding beneficial regularisation and speeding up training.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   ROOT RENDER
══════════════════════════════════════════════════════════ */
var TABS=[
  '&#128200; Sequential Boost',
  '&#129518; Residuals',
  '&#9878;&#65039; Learning Rate',
  '&#9203; Bias-Variance',
  '&#9881;&#65039; Regularisation'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.orange+','+C.red+','+C.purple+');'
    +'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Gradient Boosting</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Interactive visual walkthrough \u2014 from sequential boosting and residuals to learning rate, bias-variance and regularisation')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderIntro();
  else if(S.tab===1) html+=renderResiduals();
  else if(S.tab===2) html+=renderLearningRate();
  else if(S.tab===3) html+=renderBiasVariance();
  else if(S.tab===4) html+=renderRegularisation();
  html+='</div></div>';
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
        if(action==='tab')           {S.tab=idx;          render();}
        else if(action==='boostStep'){S.boostStep=idx;    render();}
        else if(action==='boostNext'){if(S.boostStep<4){S.boostStep++;render();}}
        else if(action==='boostPrev'){if(S.boostStep>0){S.boostStep--;render();}}
        else if(action==='residStep'){S.residStep=idx;    render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='lr')          {S.lr=val;                    render();}
        else if(action==='lrNEst') {S.lrNEst=Math.round(val);   render();}
        else if(action==='nEst')   {S.nEst=Math.round(val);     render();}
        else if(action==='maxDepth'){S.maxDepth=Math.round(val); render();}
        else if(action==='subsample'){S.subsample=val;           render();}
      });
    }
  });
}

render();
</script>
</body>
</html>"""

GB_VISUAL_HEIGHT = 1100