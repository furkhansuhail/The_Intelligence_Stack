"""
Self-contained HTML visual for Random Forests.
5 interactive tabs: Intro & Ensemble Idea, Bootstrap Sampling,
Feature Randomness, Bias-Variance & n_estimators, Feature Importance & OOB.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(RF_VISUAL_HTML, height=RF_VISUAL_HEIGHT, scrolling=True)
"""

RF_VISUAL_HTML = r"""<!DOCTYPE html>
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
  nTrees:1,        /* intro: show 1..5 trees voting */
  bootTree:0,      /* bootstrap: which tree's sample to show (0..2) */
  maxFeat:2,       /* feature randomness: sqrt subset size 1..4 */
  nEst:10,         /* bias-variance: n_estimators 1..200 */
  oobDataset:0,    /* oob tab: dataset 0 or 1 */
  impDataset:0     /* importance tab: dataset 0 or 1 */
};

/* ═══════════════════════════════════════════════════════
   SHARED DATA
═══════════════════════════════════════════════════════ */
/* 20-point 2-class dataset used across tabs */
var ALL_PTS=[
  {x:1.5,y:7.5,c:0},{x:2.2,y:8.8,c:0},{x:3.0,y:7.0,c:0},{x:1.8,y:9.2,c:0},
  {x:2.8,y:6.2,c:0},{x:1.2,y:6.8,c:0},{x:3.5,y:8.5,c:0},{x:4.2,y:7.8,c:0},
  {x:6.5,y:2.5,c:1},{x:7.8,y:3.8,c:1},{x:8.5,y:1.8,c:1},{x:7.2,y:1.2,c:1},
  {x:9.0,y:3.2,c:1},{x:6.0,y:4.5,c:1},{x:8.0,y:4.8,c:1},{x:9.5,y:2.0,c:1},
  {x:4.8,y:5.5,c:0},{x:5.5,y:4.8,c:1},{x:5.2,y:6.2,c:0},{x:6.2,y:5.8,c:1}
];

/* 5 pre-computed tree boundaries — each is a vertical split value slightly jittered */
var TREE_SPLITS=[
  {v:5.0, h:null,  side:'v'},  /* tree 1 — clean vertical */
  {v:4.6, h:5.2,   side:'v'},  /* tree 2 — left-shifted + horizontal refinement */
  {v:5.4, h:null,  side:'v'},  /* tree 3 — right-shifted */
  {v:4.9, h:4.7,   side:'v'},  /* tree 4 */
  {v:5.1, h:5.6,   side:'v'}   /* tree 5 */
];

/* classify point under tree t */
function classifyRF(p, ti){
  var sp=TREE_SPLITS[ti];
  if(sp.h===null) return p.x>sp.v?1:0;
  if(p.x>sp.v) return p.y<sp.h?1:0;
  return 0;
}

/* majority vote across first nTrees trees */
function rfVote(p, nTrees){
  var votes=[0,0];
  for(var t=0;t<nTrees;t++) votes[classifyRF(p,t)]++;
  return votes[1]>votes[0]?1:0;
}

/* ═══════════════════════════════════════════════════════
   TAB 0 — INTRO & ENSEMBLE IDEA
═══════════════════════════════════════════════════════ */
/* pre-computed per-tree colours */
var TREE_COLS=[C.blue,C.orange,C.purple,C.green,C.yellow];

function renderIntro(){
  var nT=S.nTrees;
  /* accuracy of single best tree vs ensemble */
  function acc(fn){
    return ALL_PTS.filter(function(p){return fn(p)===p.c;}).length/ALL_PTS.length;
  }
  var singleAcc=acc(function(p){return classifyRF(p,0);});
  var ensAcc=acc(function(p){return rfVote(p,nT);});

  /* ─── left SVG: ensemble boundary ─── */
  var sv=plotAxes('Feature x\u2081','Feature x\u2082');
  /* region shading from ensemble vote */
  var step=0.5;
  for(var gx=0;gx<10;gx+=step){
    for(var gy=0;gy<10;gy+=step){
      var cx=gx+step/2, cy=gy+step/2;
      var pred=rfVote({x:cx,y:cy},nT);
      sv+='<rect x="'+sx(gx).toFixed(1)+'" y="'+sy(gy+step).toFixed(1)+'"'
        +' width="'+(sx(gx+step)-sx(gx)).toFixed(1)+'" height="'+(sy(gy)-sy(gy+step)).toFixed(1)+'"'
        +' fill="'+hex(pred===0?C.blue:C.orange,0.07)+'"/>';
    }
  }
  /* draw each active tree's boundary as a faint line */
  for(var ti=0;ti<nT;ti++){
    var sp=TREE_SPLITS[ti];
    var col=TREE_COLS[ti];
    /* vertical split */
    sv+='<line x1="'+sx(sp.v).toFixed(1)+'" y1="'+PT+'" x2="'+sx(sp.v).toFixed(1)+'" y2="'+(PT+PH)+'"'
      +' stroke="'+col+'" stroke-width="'+(ti===0?2:1.2)+'" stroke-dasharray="'+(ti===0?'':'5,4')+'" opacity="0.6"/>';
    /* horizontal split if present */
    if(sp.h!==null){
      sv+='<line x1="'+sx(sp.v).toFixed(1)+'" y1="'+sy(sp.h).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(sp.h).toFixed(1)+'"'
        +' stroke="'+col+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>';
    }
  }
  /* data points */
  ALL_PTS.forEach(function(p){
    var pred=rfVote(p,nT);
    var ok=pred===p.c;
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="4.5"'
      +' fill="'+(p.c===0?C.blue:C.orange)+'" stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'"/>';
    if(!ok) sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-7).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="9">\u2717</text>';
  });
  /* accuracy badge */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="152" height="20" rx="4" fill="#0a0a0f" opacity="0.92"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+15)+'" fill="'+C.accent+'" font-size="10" font-family="monospace" font-weight="700">ensemble acc: '+(ensAcc*100).toFixed(0)+'%</text>';

  var out=sectionTitle('What is a Random Forest?','Average many diverse decision trees — each sees different data and features — to reduce variance');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* left: interactive plot */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Ensemble Boundary ('+nT+' tree'+(nT>1?'s':'')+' voting)')
    +svgBox(sv)
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;">'
    +TREE_COLS.slice(0,nT).map(function(col,i){
      return '<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
        +'<div style="width:14px;height:2px;background:'+col+';border-radius:1px;"></div>Tree '+(i+1)+'</div>';
    }).join('')
    +'</div>'
    +sliderRow('nTrees',nT,1,5,1,'n trees',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.yellow+';">\u2190 single tree</span>'
    +'<span style="color:'+C.green+';">full ensemble \u2192</span></div>'
  );
  out+='</div>';

  /* right: stats + concept */
  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT ENSEMBLE');
  var agreeCount=ALL_PTS.filter(function(p){
    var v=[];
    for(var ti2=0;ti2<nT;ti2++) v.push(classifyRF(p,ti2));
    var vote1=v.filter(function(x){return x===1;}).length;
    return vote1===nT||vote1===0;
  }).length;
  out+=statRow('Active trees',nT,C.accent);
  out+=statRow('Single tree acc',(singleAcc*100).toFixed(0)+'%',C.yellow);
  out+=statRow('Ensemble acc',(ensAcc*100).toFixed(0)+'%',C.green);
  out+=statRow('Unanimous votes',agreeCount+' / '+ALL_PTS.length,C.blue);
  out+=statRow('Split decisions',(ALL_PTS.length-agreeCount)+' / '+ALL_PTS.length,C.orange);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','THE CORE ALGORITHM');
  [
    {n:'1',lbl:'Bootstrap',desc:'Sample n examples WITH replacement',c:C.blue},
    {n:'2',lbl:'Grow tree',desc:'At each node, try only sqrt(p) random features',c:C.orange},
    {n:'3',lbl:'Repeat',desc:'Grow B independent trees on B bootstrap samples',c:C.purple},
    {n:'4',lbl:'Aggregate',desc:'Classification: majority vote\nRegression: mean',c:C.green},
  ].forEach(function(s){
    out+='<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="width:18px;height:18px;border-radius:50%;background:'+hex(s.c,0.2)+';border:1px solid '+s.c+';display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:9px;font-weight:800;color:'+s.c+';">'+s.n+'</div>'
      +'<div><div style="font-size:9.5px;font-weight:700;color:'+s.c+';">'+s.lbl+'</div>'
      +'<div style="font-size:8.5px;color:'+C.muted+';margin-top:1px;">'+s.desc+'</div></div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHY BETTER THAN ONE TREE?');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'Single tree: <span style="color:'+C.red+';font-weight:700;">high variance</span> \u2014 small data changes \u2192 very different tree.<br><br>'
    +'Average of B trees: <span style="color:'+C.green+';font-weight:700;">variance \u2248 \u03c3\u00b2/B</span> if trees are independent.<br><br>'
    +'Bootstrap + feature randomness <span style="color:'+C.yellow+';font-weight:700;">decorrelates trees</span> so errors partially cancel out.'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127795;','The Wisdom of Diverse Crowds',
    'A Random Forest works because trees make <span style="color:'+C.yellow+';font-weight:700;">different errors</span>. '
    +'If all trees were identical (trained on the same data with the same features), '
    +'averaging would do nothing. '
    +'Bootstrap sampling and random feature subsets ensure each tree is a <span style="color:'+C.accent+';font-weight:700;">different view</span> of the data — '
    +'their majority vote cancels individual mistakes. '
    +'Add trees to the slider and watch the boundary <span style="color:'+C.green+';font-weight:700;">stabilise</span>.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 1 — BOOTSTRAP SAMPLING
═══════════════════════════════════════════════════════ */
/* 3 deterministic bootstrap samples (indices into ALL_PTS, with replacement) */
var BOOT_SAMPLES=[
  [0,2,0,3,5,7,1,8,11,9,14,12,16,19,17,18,4,6,10,13],  /* tree 1 */
  [1,1,3,4,6,0,8,9,9,10,13,15,17,16,18,19,2,5,11,12],  /* tree 2 */
  [2,3,5,5,6,7,0,8,10,11,11,12,15,14,16,17,18,19,1,4]   /* tree 3 */
];

function renderBootstrap(){
  var ti=S.bootTree;
  var sample=BOOT_SAMPLES[ti];
  var inBag=[];
  for(var i=0;i<20;i++) inBag.push(false);
  sample.forEach(function(idx){inBag[idx]=true;});

  /* count duplicates in bootstrap */
  var counts=[];
  for(var i=0;i<20;i++) counts.push(0);
  sample.forEach(function(idx){counts[idx]++;});

  var oobPts=ALL_PTS.filter(function(p,i){return !inBag[i];});
  var oobCount=oobPts.length;
  var avgDupes=(sample.filter(function(x,i){return sample.indexOf(x)!==i;}).length);

  /* ─── SVG: show in-bag vs OOB points ─── */
  var sv=plotAxes('Feature x\u2081','Feature x\u2082');
  /* tree boundary */
  var sp=TREE_SPLITS[ti];
  sv+='<line x1="'+sx(sp.v).toFixed(1)+'" y1="'+PT+'" x2="'+sx(sp.v).toFixed(1)+'" y2="'+(PT+PH)+'"'
    +' stroke="'+TREE_COLS[ti]+'" stroke-width="2" opacity="0.5" stroke-dasharray="6,3"/>';

  ALL_PTS.forEach(function(p,i){
    var cnt=counts[i];
    var inb=inBag[i];
    var col=p.c===0?C.blue:C.orange;
    if(!inb){
      /* OOB point — grey ring, no fill */
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5"'
        +' fill="'+hex(C.dim,0.4)+'" stroke="'+C.muted+'" stroke-width="1.5" stroke-dasharray="3,2"/>';
      sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)+4).toFixed(1)+'" text-anchor="middle" fill="'+C.muted+'" font-size="7">OOB</text>';
    } else {
      /* in-bag — solid, with duplicate count ring if cnt>1 */
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="'+(4+cnt)+'"'
        +' fill="none" stroke="'+col+'" stroke-width="'+(cnt>1?1.5:0)+'" opacity="0.4"/>';
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="4.5"'
        +' fill="'+col+'" stroke="#0a0a0f" stroke-width="1.5"/>';
      if(cnt>1){
        sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-8).toFixed(1)+'" text-anchor="middle"'
          +' fill="'+C.yellow+'" font-size="8" font-weight="700">\u00d7'+cnt+'</text>';
      }
    }
  });
  /* badge */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="148" height="20" rx="4" fill="#0a0a0f" opacity="0.92"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+15)+'" fill="'+TREE_COLS[ti]+'" font-size="10" font-family="monospace" font-weight="700">Tree '+(ti+1)+' bootstrap sample</text>';

  /* ─── right: OOB prediction matrix (simplified) ─── */
  var out=sectionTitle('Bootstrap Sampling','Each tree trains on ~63% of data — the remaining ~37% are Out-Of-Bag (OOB) for free validation');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
  [0,1,2].forEach(function(i){
    out+=btnSel(i,ti,TREE_COLS[i],'Tree '+(i+1),'bootTree');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Tree '+(ti+1)+': In-bag vs Out-of-Bag')
    +svgBox(sv)
    +'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;">'
    +[
      {col:C.blue,  lbl:'Class 0 in-bag'},
      {col:C.orange,lbl:'Class 1 in-bag'},
      {col:C.muted, lbl:'OOB (unused)',ring:true},
      {col:C.yellow,lbl:'\u00d7N = duplicated'},
    ].map(function(it){
      return '<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
        +'<div style="width:10px;height:10px;border-radius:50%;background:'+(it.ring?'transparent':it.col)
        +(it.ring?';border:1.5px dashed '+it.col:'')+'"></div>'+it.lbl+'</div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','TREE '+(ti+1)+' STATISTICS');
  out+=statRow('Training set size',sample.length,C.accent);
  out+=statRow('Unique samples',ALL_PTS.filter(function(p,i){return inBag[i];}).length,C.blue);
  out+=statRow('Duplicate draws',avgDupes,C.yellow);
  out+=statRow('OOB samples',oobCount,C.orange);
  out+=statRow('OOB fraction',(oobCount/ALL_PTS.length*100).toFixed(0)+'%',C.purple);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHY ~63% IN-BAG?');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'Probability a sample is <em>not</em> drawn in n draws:<br>'
    +'<span style="color:'+C.accent+';font-family:monospace;">(1 \u2212 1/n)\u207f \u2192 e\u207b\u00b9 \u2248 0.368</span><br><br>'
    +'So ~36.8% are OOB per tree \u2014 a free validation set. '
    +'No cross-validation needed to estimate generalisation error.'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','OOB ERROR ESTIMATE');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'For each training sample x\u1d35:<br>'
    +'1. Find all trees where x\u1d35 was <span style="color:'+C.orange+';">OOB</span><br>'
    +'2. Average <em>only those trees\'</em> predictions<br>'
    +'3. Compare to true label y\u1d35<br><br>'
    +'Aggregate across all samples \u2192 <span style="color:'+C.green+';font-weight:700;">OOB score</span>. '
    +'Correlates strongly with test-set performance.'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127922;','Sampling With Replacement = Diversity',
    'Each bootstrap sample draws n points with replacement from n training points. '
    +'On average each tree sees <span style="color:'+C.green+';font-weight:700;">~63.2%</span> of training data. '
    +'The <span style="color:'+C.orange+';font-weight:700;">~36.8% OOB</span> points are invisible during training — '
    +'making them a free, unbiased validation set. '
    +'sklearn exposes this as <span style="color:'+C.accent+';font-family:monospace;">oob_score=True</span>.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 2 — FEATURE RANDOMNESS
═══════════════════════════════════════════════════════ */
var ALL_FEATURES=['Age','Income','Height','Weight','Score','Rating','Count','Rate'];
/* pre-set random subsets for each node demonstration */
var FEAT_SUBSETS=[
  [[0,3,6],[1,4,7],[0,2,5],[2,3,7]],  /* max_features=1 */
  [[0,3],[1,5],[2,4],[3,6]],           /* placeholder — overridden dynamically */
  [[0,2,4],[1,3,5],[0,4,6],[2,5,7]],  /* max_features=3 */
  [[0,1,2,4],[1,2,5,6],[0,3,4,7],[1,4,5,6]] /* max_features=4 */
];
/* hard-coded correlation/gain values for each subset at each node */
var NODE_GAINS=[
  [0.32,0.28,0.41,0.19],
  [0.38,0.22,0.44,0.31],
  [0.29,0.40,0.18,0.35],
  [0.42,0.31,0.27,0.39]
];

function renderFeatureRandom(){
  var mf=S.maxFeat; /* 1..4 */
  var p=ALL_FEATURES.length;
  var sqrtP=Math.round(Math.sqrt(p)); /* =3 for p=8 */

  /* Build 4-node tree diagram showing feature subsets */
  /* Node positions */
  var nodes=[
    {id:'root',x:220,y:20,label:'Root node',depth:0},
    {id:'l1',  x:120,y:90,label:'Node L',  depth:1},
    {id:'r1',  x:320,y:90,label:'Node R',  depth:1},
    {id:'l2',  x:60, y:160,label:'Node LL', depth:2},
    {id:'r2',  x:180,y:160,label:'Node LR', depth:2}
  ];
  var edges=[
    {from:0,to:1,label:'Yes'},{from:0,to:2,label:'No'},
    {from:1,to:3,label:'Yes'},{from:1,to:4,label:'No'}
  ];

  /* feature subset for each node based on mf */
  function getSubset(nodeIdx){
    /* generate a deterministic subset of size mf from ALL_FEATURES */
    var seed=nodeIdx*7+mf*3;
    var idxs=[];
    for(var k=0;k<p;k++) idxs.push(k);
    /* pseudo-shuffle with seed */
    for(var i2=idxs.length-1;i2>0;i2--){
      var j=(seed*31+i2*17)%( i2+1);
      var tmp=idxs[i2]; idxs[i2]=idxs[j]; idxs[j]=tmp;
    }
    return idxs.slice(0,mf);
  }

  /* Draw tree SVG */
  var TW=440, TH=220;
  var tv='';
  /* edges */
  edges.forEach(function(e){
    var fn=nodes[e.from], tn=nodes[e.to];
    var x1=fn.x, y1=fn.y+28, x2=tn.x, y2=tn.y;
    var mx=(x1+x2)/2, my=(y1+y2)/2;
    tv+='<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
    tv+='<rect x="'+(mx-18)+'" y="'+(my-7)+'" width="36" height="13" rx="3" fill="#0a0a0f" opacity="0.9"/>';
    tv+='<text x="'+mx+'" y="'+(my+3)+'" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace">'+e.label+'</text>';
  });
  /* nodes */
  nodes.forEach(function(n,ni){
    var sub=getSubset(ni);
    var gainIdx=ni<4?ni:3;
    var gain=NODE_GAINS[Math.min(mf-1,3)][gainIdx];
    var bestFeat=ALL_FEATURES[sub[0]];
    var col=ni===0?C.accent:(ni<3?C.blue:C.purple);
    tv+='<rect x="'+(n.x-68)+'" y="'+n.y+'" width="136" height="28" rx="6" fill="'+hex(col,0.1)+'" stroke="'+col+'" stroke-width="1.5"/>';
    tv+='<text x="'+n.x+'" y="'+(n.y+12)+'" text-anchor="middle" fill="'+col+'" font-size="9" font-weight="700">'+n.label+'</text>';
    tv+='<text x="'+n.x+'" y="'+(n.y+23)+'" text-anchor="middle" fill="'+C.muted+'" font-size="7.5">'
      +'try: ['+sub.map(function(s){return ALL_FEATURES[s];}).join(', ')+']\u2192 '+bestFeat+'</text>';
  });

  /* ─── feature importance bar inside this tree ─── */
  /* count how often each feature appears in subsets */
  var featCounts=[];
  for(var fi=0;fi<p;fi++) featCounts.push(0);
  nodes.forEach(function(n,ni){
    getSubset(ni).forEach(function(si){ featCounts[si]++; });
  });
  var maxCnt=Math.max.apply(null,featCounts);

  /* correlation matrix for corr reduction */
  var corrSingle=[0.82,0.76,0.89,0.71,0.85,0.78,0.92,0.68]; /* corr with best single-tree */
  var corrRF=corrSingle.map(function(v){ return Math.max(0.1, v - (mf/p)*0.55); });

  var out=sectionTitle('Feature Randomness (max_features)','At each split, only a random subset of features is considered — this decorrelates the trees');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Feature Subsets at Each Node (max_features='+mf+')')
    +svgBox(tv,TW,TH)
    +sliderRow('maxFeat',mf,1,4,1,'max_features',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span>\u2190 1 feat (max randomness)</span>'
    +'<span>all '+p+' feats (no randomness) \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','SETTINGS (p='+p+' features)');
  var defaultSqrt='\u221a'+p+' \u2248 '+sqrtP;
  out+=statRow('max_features',mf,C.accent);
  out+=statRow('sklearn default (clf)',defaultSqrt,C.yellow);
  out+=statRow('sklearn default (reg)','\u230ap/3\u230b = '+Math.floor(p/3),C.blue);
  out+=statRow('Fraction considered',(mf/p*100).toFixed(0)+'%',C.purple);
  out+=statRow('Splits evaluated',mf+' (vs '+p+' all)',C.green);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','INTER-TREE CORRELATION (\u03c1)');
  out+=div('font-size:8.5px;color:'+C.muted+';margin-bottom:8px;line-height:1.6;',
    'If two trees always use the same strong feature at the root, '
    +'they make similar errors \u2014 ensemble variance = '
    +'<span style="color:'+C.red+';">\u03c1\u03c3\u00b2</span> not '
    +'<span style="color:'+C.green+';">\u03c3\u00b2/B</span>.'
  );
  ['F1','F2','F3','F4'].forEach(function(fname,fi){
    var rho=corrRF[fi];
    var barW=Math.round(rho*100);
    var col=rho>0.6?C.red:rho>0.35?C.yellow:C.green;
    out+='<div style="margin:3px 0;">'
      +'<div style="display:flex;justify-content:space-between;font-size:8.5px;margin-bottom:2px;">'
      +'<span style="color:'+C.muted+';">'+fname+'</span>'
      +'<span style="color:'+col+';font-weight:700;">\u03c1='+(rho).toFixed(2)+'</span></div>'
      +'<div style="height:6px;border-radius:3px;background:'+C.border+';">'
      +'<div style="height:6px;border-radius:3px;width:'+barW+'%;background:'+col+';"></div>'
      +'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','max_features TRADEOFF');
  [
    {lbl:'Too small (1)',pro:'Max decorrelation',con:'Each tree: high bias',c:C.orange},
    {lbl:'sqrt(p) (default)',pro:'Good bias/corr balance',con:'Standard choice',c:C.green},
    {lbl:'All p (=bagging)',pro:'Best individual trees',con:'High inter-tree corr',c:C.red},
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
      +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+r.pro+'</div>'
      +'<div style="font-size:8px;color:'+C.red+';">\u2718 '+r.con+'</div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9855;','Variance = Correlation \xd7 Tree Variance',
    'Ensemble variance = <span style="color:'+C.accent+';font-family:monospace;">\u03c1 \u00b7 \u03c3\u00b2 + (1\u2212\u03c1) \u00b7 \u03c3\u00b2/B</span>. '
    +'As B\u2192\u221e the second term vanishes, leaving <span style="color:'+C.red+';font-family:monospace;">\u03c1 \u00b7 \u03c3\u00b2</span>. '
    +'<span style="color:'+C.yellow+';font-weight:700;">Reducing \u03c1</span> (via random features) is what makes RF better than plain bagging. '
    +'This is the key mathematical insight behind Breiman (2001).'
  );
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 3 — BIAS-VARIANCE & n_estimators
═══════════════════════════════════════════════════════ */
function rfTestErr(n){
  /* simulated: rapidly drops then plateaus */
  return 0.07 + 0.20*Math.exp(-0.09*n) + 0.005*(Math.random?0:0);
}
function rfTrainErr(n){
  return 0.01 + 0.01*Math.exp(-0.05*n);
}
function rfOOBErr(n){
  return 0.075 + 0.22*Math.exp(-0.08*n);
}
function singleTreeErr(){return 0.19;}

function renderBiasVariance(){
  var nE=S.nEst;
  var trE=rfTrainErr(nE), teE=rfTestErr(nE), oobE=rfOOBErr(nE);
  var plateau=0.075;

  /* ─── error curve plot ─── */
  var sv=plotAxes('n_estimators','Error',200,0.3,
    [0,25,50,75,100,125,150,175,200],
    [0,0.05,0.10,0.15,0.20,0.25,0.30]);

  /* single tree baseline */
  var stErr=singleTreeErr();
  sv+='<line x1="'+PL+'" y1="'+sy(stErr,0.3).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(stErr,0.3).toFixed(1)+'"'
    +' stroke="'+C.yellow+'" stroke-width="1" stroke-dasharray="6,4" opacity="0.6"/>';
  sv+='<text x="'+(PL+PW-4)+'" y="'+(sy(stErr,0.3)-4).toFixed(1)+'" text-anchor="end" fill="'+C.yellow+'" font-size="8" font-family="monospace">single tree</text>';

  /* plateau */
  sv+='<line x1="'+PL+'" y1="'+sy(plateau,0.3).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(plateau,0.3).toFixed(1)+'"'
    +' stroke="'+C.green+'" stroke-width="1" stroke-dasharray="3,3" opacity="0.4"/>';
  sv+='<text x="'+(PL+4)+'" y="'+(sy(plateau,0.3)-4).toFixed(1)+'" fill="'+C.green+'" font-size="8">plateau</text>';

  /* curve paths */
  var trainPath='', testPath='', oobPath='';
  var ns=[1,2,3,5,8,12,18,25,35,50,70,100,140,200];
  ns.forEach(function(n,i){
    var px=sx(n,200);
    var pyt=sy(rfTrainErr(n),0.3), pyte=sy(rfTestErr(n),0.3), pyob=sy(rfOOBErr(n),0.3);
    trainPath+=(i===0?'M':'L')+px.toFixed(1)+','+pyt.toFixed(1)+' ';
    testPath+=(i===0?'M':'L')+px.toFixed(1)+','+pyte.toFixed(1)+' ';
    oobPath+=(i===0?'M':'L')+px.toFixed(1)+','+pyob.toFixed(1)+' ';
  });
  sv+='<path d="'+trainPath+'" fill="none" stroke="'+C.green+'" stroke-width="2"/>';
  sv+='<path d="'+testPath+'" fill="none" stroke="'+C.red+'" stroke-width="2"/>';
  sv+='<path d="'+oobPath+'" fill="none" stroke="'+C.orange+'" stroke-width="2" stroke-dasharray="5,3"/>';

  /* current n_est cursor */
  var cvx=sx(nE,200);
  sv+='<line x1="'+cvx.toFixed(1)+'" y1="'+PT+'" x2="'+cvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+sy(trE,0.3).toFixed(1)+'" r="4" fill="'+C.green+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+sy(teE,0.3).toFixed(1)+'" r="4" fill="'+C.red+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+sy(oobE,0.3).toFixed(1)+'" r="4" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5"/>';

  /* legend */
  sv+='<rect x="'+(PL+PW-110)+'" y="'+(PT+3)+'" width="108" height="54" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  [
    {c:C.green, l:'Train error'},
    {c:C.red,   l:'Test error'},
    {c:C.orange,l:'OOB error',dash:true},
    {c:C.yellow,l:'Single tree',dash:true},
  ].forEach(function(item,li){
    var ly2=PT+13+li*11;
    sv+='<line x1="'+(PL+PW-106)+'" y1="'+ly2+'" x2="'+(PL+PW-86)+'" y2="'+ly2+'" stroke="'+item.c+'" stroke-width="2"'+(item.dash?' stroke-dasharray="4,2"':'')+'/>';
    sv+='<text x="'+(PL+PW-82)+'" y="'+(ly2+3)+'" fill="'+item.c+'" font-size="8" font-family="monospace">'+item.l+'</text>';
  });

  var regime=nE<10?'Underfitting':nE<50?'Converging':'Converged';
  var regimeCol=nE<10?C.orange:nE<50?C.yellow:C.green;
  var overfitting=nE>100?'No (RF never overfits via n_estimators)':'No';

  var out=sectionTitle('Bias-Variance & n_estimators','More trees always reduce variance — unlike depth, n_estimators never causes overfitting');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Error vs n_estimators')
    +svgBox(sv)
    +sliderRow('nEst',nE,1,200,1,'n_estimators',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.orange+';">\u2190 few trees (high variance)</span>'
    +'<span style="color:'+C.green+';">converged \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','AT n_estimators='+nE);
  out+=statRow('Train error',(trE*100).toFixed(1)+'%',C.green);
  out+=statRow('Test error',(teE*100).toFixed(1)+'%',C.red);
  out+=statRow('OOB error',(oobE*100).toFixed(1)+'%',C.orange);
  out+=statRow('Single tree err',(stErr*100).toFixed(1)+'%',C.yellow);
  out+=statRow('Improvement',((stErr-teE)*100).toFixed(1)+'pp vs single',C.accent);
  out+=statRow('Status',regime,regimeCol);
  out+=statRow('Overfitting risk',overfitting,C.green);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHY NO OVERFIT?');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'Adding trees averages predictions \u2014 it does <em>not</em> add parameters to the model. '
    +'Bias is set by individual tree depth. '
    +'Variance <span style="color:'+C.green+';">\u2265 \u03c1\u03c3\u00b2</span> as B\u2192\u221e. '
    +'More trees can only help or be neutral \u2014 never hurt. '
    +'The trade-off is purely <span style="color:'+C.yellow+';">compute time</span>, not accuracy.'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PRACTICAL GUIDANCE');
  [
    {lbl:'n_estimators',          val:'100\u2013500',        c:C.accent},
    {lbl:'max_depth',             val:'None (full trees)',   c:C.yellow},
    {lbl:'min_samples_leaf',      val:'1\u20135',            c:C.blue},
    {lbl:'max_features',          val:'"sqrt" (classifier)', c:C.purple},
    {lbl:'n_jobs',                val:'-1 (all cores)',      c:C.green},
    {lbl:'oob_score',             val:'True (free metric)',  c:C.orange},
  ].forEach(function(r){
    out+='<div style="display:flex;justify-content:space-between;font-size:9px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<span style="color:'+C.muted+';font-family:monospace;">'+r.lbl+'</span>'
      +'<span style="color:'+r.c+';font-weight:700;">'+r.val+'</span></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#8734;','More Trees = Free Lunch (Until Compute)',
    '<span style="color:'+C.red+';font-weight:700;">Depth</span> controls bias and can overfit. '
    +'<span style="color:'+C.green+';font-weight:700;">n_estimators</span> only reduces variance \u2014 it cannot overfit. '
    +'Error curves <span style="color:'+C.accent+';font-weight:700;">asymptotically plateau</span> around n=100\u2013300 for most datasets. '
    +'Use <span style="color:'+C.yellow+';font-family:monospace;">oob_score=True</span> to cheaply monitor convergence without a hold-out set.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════════
   TAB 4 — FEATURE IMPORTANCE & OOB SCORE
═══════════════════════════════════════════════════════ */
var IMP_DATASETS=[
  {
    name:'Forest Fire Risk',
    features:['Temperature','Humidity','Wind Speed','Drought Index','Vegetation','Slope','Aspect','Elevation'],
    /* per-tree importances (3 trees) to show averaging */
    treeImps:[
      [0.31,0.24,0.18,0.12,0.07,0.04,0.02,0.02],
      [0.28,0.26,0.16,0.15,0.06,0.05,0.03,0.01],
      [0.34,0.22,0.20,0.10,0.07,0.04,0.02,0.01]
    ],
    cols:[C.red,C.blue,C.accent,C.orange,C.green,C.yellow,C.purple,C.muted],
    oobScore:0.871,
    note:'Temperature dominates across all trees \u2014 stable importance signal.'
  },
  {
    name:'Credit Default',
    features:['Debt Ratio','Credit Score','Income','Loan Amount','Missed Payments','Account Age','Utilisation','Dependents'],
    treeImps:[
      [0.35,0.28,0.14,0.10,0.06,0.04,0.02,0.01],
      [0.30,0.31,0.16,0.08,0.07,0.04,0.03,0.01],
      [0.38,0.25,0.12,0.12,0.05,0.04,0.03,0.01]
    ],
    cols:[C.orange,C.green,C.blue,C.red,C.yellow,C.purple,C.accent,C.muted],
    oobScore:0.913,
    note:'Debt Ratio and Credit Score most important \u2014 consistent across all 3 trees.'
  }
];

function renderImportance(){
  var ds=IMP_DATASETS[S.impDataset];
  /* average importances across trees */
  var avgImps=ds.features.map(function(f,fi){
    return ds.treeImps.reduce(function(s,t){return s+t[fi];},0)/ds.treeImps.length;
  });
  var total=avgImps.reduce(function(a,b){return a+b;},0);
  var norm=avgImps.map(function(v){return v/total;});

  /* bar chart */
  var BW=420, BH=210;
  var bPL=105, bPR=20, bPT=15, bPB=22;
  var bPW=BW-bPL-bPR, bPH=BH-bPT-bPB;
  var barH=Math.floor(bPH/norm.length)-5;

  var sv2='';
  /* grid */
  [0,0.1,0.2,0.3,0.4].forEach(function(v){
    var bx=bPL+v*bPW;
    sv2+='<line x1="'+bx.toFixed(1)+'" y1="'+bPT+'" x2="'+bx.toFixed(1)+'" y2="'+(bPT+bPH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv2+='<text x="'+bx.toFixed(1)+'" y="'+(bPT+bPH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+(v*100).toFixed(0)+'%</text>';
  });
  sv2+='<line x1="'+bPL+'" y1="'+bPT+'" x2="'+bPL+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv2+='<text x="'+(bPL+bPW/2)+'" y="'+(BH-3)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">Avg MDI Importance (%) across '+ds.treeImps.length+' trees</text>';

  norm.forEach(function(v,i){
    var by=bPT+i*(barH+5);
    var bw=v*bPW;
    var col=ds.cols[i];
    /* per-tree spread: thin lines */
    ds.treeImps.forEach(function(tree){
      var tv2=tree[i]/total*bPW;
      sv2+='<line x1="'+(bPL+tv2).toFixed(1)+'" y1="'+(by+2)+'" x2="'+(bPL+tv2).toFixed(1)+'" y2="'+(by+barH-2)+'"'
        +' stroke="'+col+'" stroke-width="1.5" opacity="0.35"/>';
    });
    sv2+='<rect x="'+bPL+'" y="'+by+'" width="'+bPW+'" height="'+barH+'" rx="3" fill="'+hex(col,0.08)+'"/>';
    sv2+='<rect x="'+bPL+'" y="'+by+'" width="'+bw.toFixed(1)+'" height="'+barH+'" rx="3" fill="'+hex(col,0.72)+'"/>';
    if(i===0){
      sv2+='<rect x="'+bPL+'" y="'+by+'" width="'+bw.toFixed(1)+'" height="'+barH+'" rx="3" fill="none" stroke="'+col+'" stroke-width="1.5"/>';
    }
    sv2+='<text x="'+(bPL-5)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="'+(i<3?C.text:C.muted)+'" font-size="8.5" font-family="monospace">'+ds.features[i]+'</text>';
    if(bw>22){
      sv2+='<text x="'+(bPL+bw-3)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="#0a0a0f" font-size="8" font-weight="700">'+(v*100).toFixed(1)+'%</text>';
    }
  });
  /* formula box */
  var fx=bPL+bPW*0.56, fy=bPT+6;
  sv2+='<rect x="'+fx.toFixed(1)+'" y="'+fy.toFixed(1)+'" width="140" height="28" rx="4" fill="#0a0a0f" opacity="0.92" stroke="'+C.border+'"/>';
  sv2+='<text x="'+(fx+70).toFixed(1)+'" y="'+(fy+10).toFixed(1)+'" text-anchor="middle" fill="'+C.muted+'" font-size="7.5">RF Imp(f) = avg over B trees</text>';
  sv2+='<text x="'+(fx+70).toFixed(1)+'" y="'+(fy+21).toFixed(1)+'" text-anchor="middle" fill="'+C.accent+'" font-size="7.5">(1/B)\u03a3 tree\u1d47 Imp\u1d47(f)</text>';

  var out=sectionTitle('Feature Importance & OOB Score','RF aggregates MDI across all trees for stable importance; OOB error is a free generalisation estimate');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
  IMP_DATASETS.forEach(function(d,i){
    out+=btnSel(i,S.impDataset,C.accent,'&#128202; '+d.name,'impData');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',ds.name+' \u2014 RF Feature Importances')
    +svgBox(sv2,BW,BH)
    +'<div style="margin-top:8px;padding:7px 10px;background:#08080d;border-radius:6px;border:1px solid '+C.border+';">'
    +'<div style="font-size:8px;color:'+C.muted+';line-height:1.6;">&#124;&#124; = per-tree importance spread &nbsp;&nbsp; bar = forest average &nbsp;&nbsp; '+ds.note+'</div>'
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','OOB SCORE');
  var oobCol=ds.oobScore>0.9?C.green:ds.oobScore>0.8?C.yellow:C.orange;
  out+=statRow('OOB accuracy',(ds.oobScore*100).toFixed(1)+'%',oobCol);
  out+=statRow('OOB error',((1-ds.oobScore)*100).toFixed(1)+'%',C.red);
  out+=statRow('Trees used','100',C.accent);
  out+=statRow('Free CV equivalent','~10-fold',C.blue);
  out+=statRow('Extra cost','~0 (uses OOB preds)',C.green);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','RF vs SINGLE TREE IMPORTANCE');
  [
    {lbl:'Stability',  rf:C.green,  dt:C.red,   rfv:'High (avg B trees)',  dtv:'Low (1 tree)'},
    {lbl:'Bias',       rf:C.yellow, dt:C.orange, rfv:'MDI bias remains',   dtv:'Same MDI bias'},
    {lbl:'Permutation',rf:C.green,  dt:C.green,  rfv:'Also available',     dtv:'Also available'},
    {lbl:'Speed',      rf:C.orange, dt:C.green,  rfv:'B\u00d7 slower',     dtv:'Fast'},
  ].forEach(function(r){
    out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="flex:0.9;color:'+C.muted+';">'+r.lbl+'</div>'
      +'<div style="flex:1;color:'+r.rf+';">&#127794; '+r.rfv+'</div>'
      +'</div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','BEYOND MDI');
  [
    {lbl:'Permutation Imp.', desc:'shuffle feature \u2192 measure OOB drop',  c:C.blue},
    {lbl:'SHAP values',      desc:'per-prediction attribution, direction + magnitude', c:C.purple},
    {lbl:'Partial dependence',desc:'marginal effect of one feature on output', c:C.yellow},
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.desc+'</div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128202;','MDI is More Stable in a Forest',
    'A single tree\'s MDI is noisy \u2014 one different split at the root changes all downstream importances. '
    +'Averaging across B trees <span style="color:'+C.green+';font-weight:700;">smooths this variance</span>. '
    +'The thin vertical lines on the bars show per-tree spread. '
    +'For unbiased importance, use <span style="color:'+C.accent+';font-family:monospace;">permutation_importance</span> '
    +'(measures OOB score drop when feature values are shuffled). '
    +'For direction and interaction effects, <span style="color:'+C.yellow+';font-weight:700;">SHAP TreeExplainer</span> is the gold standard.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════════
   ROOT RENDER
═══════════════════════════════════════════════════════ */
var TABS=[
  '&#127795; Intro &amp; Ensemble',
  '&#127922; Bootstrap Sampling',
  '&#9855; Feature Randomness',
  '&#8734; Bias-Variance',
  '&#128202; Importance &amp; OOB'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.green+','+C.accent+','+C.blue+');'
    +'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Random Forest</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Interactive visual walkthrough \u2014 from ensemble voting and bootstrap sampling to feature randomness, bias-variance and OOB score')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderIntro();
  else if(S.tab===1) html+=renderBootstrap();
  else if(S.tab===2) html+=renderFeatureRandom();
  else if(S.tab===3) html+=renderBiasVariance();
  else if(S.tab===4) html+=renderImportance();
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
        if(action==='tab')           {S.tab=idx;       render();}
        else if(action==='bootTree') {S.bootTree=idx;  render();}
        else if(action==='impData')  {S.impDataset=idx;render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='nTrees')       {S.nTrees=Math.round(val);  render();}
        else if(action==='maxFeat') {S.maxFeat=Math.round(val); render();}
        else if(action==='nEst')    {S.nEst=Math.round(val);    render();}
      });
    }
  });
}

render();
</script>
</body>
</html>"""

RF_VISUAL_HEIGHT = 1100