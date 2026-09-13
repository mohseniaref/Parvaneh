"""Build teaching notebooks; execute with nbconvert before committing outputs.

Requires nbformat (part of Jupyter). No external images or datasets are used.
Run from the repository root. Regeneration clears stored notebook outputs.
"""
from pathlib import Path
from textwrap import dedent
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT = Path.cwd().resolve()
if not (ROOT / 'src' / 'parvaneh').is_dir(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import parvaneh as pv
plt.rcParams.update({'figure.dpi': 115, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
def W(x): return (np.asarray(x)+np.pi) % (2*np.pi)-np.pi
def table(headers, rows):
    display(Markdown('| '+' | '.join(headers)+' |\\n|'+
        '|'.join(['---']*len(headers))+'|\\n'+
        '\\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)))
def images(arrays, titles, cmap='viridis'):
    fig,axes=plt.subplots(1,len(arrays),figsize=(4*len(arrays),3.4),constrained_layout=True)
    for ax,a,title in zip(np.atleast_1d(axes),arrays,titles):
        im=ax.imshow(a,cmap=cmap); ax.set(title=title,xlabel='column',ylabel='row')
        fig.colorbar(im,ax=ax,shrink=.75)
    plt.show()
def align(u,t): return u-np.mean(u-t)
def scene():
    y,x=np.mgrid[0:18,0:22]
    truth=.38*x+.22*y+1.4*np.exp(-((x-12)**2+(y-8)**2)/25)
    patch=(x>=8)&(x<=14)&(y>=5)&(y<=12)
    rng=np.random.default_rng(21)
    phase=W(truth+rng.normal(size=truth.shape)*np.where(patch,1.35,.06))
    confidence=np.where(patch,.2,1.)
    return truth,phase,confidence
def report(u,t,w):
    assert u.shape==t.shape and np.isfinite(u).all()
    table(['Quantity','Measured value'],[
        ['Aligned RMSE (rad)',f'{pv.rmse_aligned(u,t):.6f}'],
        ['Raw max wrapped difference from input (rad)',f'{np.max(np.abs(W(u-w))):.6f}']])
    images([align(u,t),align(u,t)-t],['Aligned reconstruction (rad)','Error against truth (rad)'])
'''

def lesson(name,title,question,theory,demo,interpret,run,experiment,limits,module):
    cells=[]
    def md(s): cells.append(nb.v4.new_markdown_cell(dedent(s).strip()))
    def code(s): cells.append(nb.v4.new_code_cell(dedent(s).strip()))
    md('# '+title+'\n\n'+question+'\n\n'
       'We start with numbers that can be checked by hand, draw the algorithm’s decision, '
       'then run the public API on data with known truth. All phase values are in radians unless stated otherwise.')
    code(SETUP)
    md('## 1 · Work through the model\n\n'+dedent(theory).strip())
    code(demo)
    md('## 2 · Read the calculation\n\n'+dedent(interpret).strip())
    md('## 3 · Run the implementation\n\n'
       'The seed and input sizes below are fixed. A low objective or a converged solver '
       'means the chosen mathematical problem was solved; it does not prove the physical phase was recovered.')
    code(run)
    md('## 4 · Change one choice and measure its effect\n\n'
       'This controlled experiment changes a solver setting, data condition, or reference calculation. '
       'Compare the measured values before drawing a conclusion.')
    code(experiment)
    md('## 5 · Interpretation and limits\n\n'+dedent(limits).strip())
    md('## Continue exploring\n\n'
       'Rerun with another seed or a larger phase step. Predict which assumption fails first, '
       'then inspect the result and report. The method’s output convention and masks matter '
       'as much as its RMSE.\n\n'
       f'Implementation: [`{module}.py`](../src/parvaneh/{module}.py). '
       'See the [API reference](../docs/api.md), [notebook index](../docs/notebooks.md), '
       '[mathematical derivations](../docs/mathematics.md), and '
       '[bibliography](../docs/references.md) for further reading. These examples describe '
       'the current implementation, including its limitations; they are not claims of numerical '
       'identity with external software.')
    out=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'display_name':'Python 3','language':'python','name':'python3'}})
    nb.validate(out); nb.write(out,ROOT/'notebooks'/name)

lesson('least_squares_and_weights.ipynb','Least squares: solve a four-pixel disagreement',
       'How can one surface approximate neighbor differences that disagree around a loop?',r'''
Label a square in row-major order $A,B,D,C$. Orient its edges $A\to B$,
$B\to C$, $D\to C$, $A\to D$. Each row of $D$ contains $-1$ at the tail
and $+1$ at the head. Thus $Du$ is the vector of ordinary phase differences.

Take $w/\pi=(0,0.6,-0.2,-0.8)$. Wrapping each difference gives
$g/\pi=(0.6,0.6,-0.6,-0.2)$, and the clockwise sum is $2\pi$.
No $u$ satisfies all four measurements exactly.

1. Form residuals $e=Du-g$.
2. Minimize $J=e^Te$; differentiate: $\nabla J=2D^T(Du-g)$.
3. Solve $D^TDu=D^Tg$ with $\sum u=0$ to fix the nullspace.

$D^TD$ is the **positive graph Laplacian**, corresponding to minus the usual
continuous Laplacian. With divergence $=-D^T$, the equivalent Poisson form is
$\Delta u=\operatorname{div}g$. Keeping the signs consistent avoids a common mistake.
''','''
D=np.array([[-1,1,0,0],[0,-1,0,1],[0,0,-1,1],[-1,0,1,0]],float)
w=np.pi*np.array([0,.6,-.2,-.8]); g=W(D@w)
u=np.linalg.lstsq(np.vstack([D,np.ones(4)]),np.r_[g,0],rcond=None)[0]
table(['Edge','Measured / π','Fitted / π','Residual / π'],
      [(e,f'{a/np.pi:.2f}',f'{b/np.pi:.2f}',f'{(b-a)/np.pi:.2f}')
       for e,a,b in zip(['AB','BC','DC','AD'],g,D@u)])
assert np.allclose(D.T@(D@u-g),0)
images([w.reshape(2,2)/np.pi,u.reshape(2,2)/np.pi],['Measured phase / π','Least-squares phase / π'])
''',r'''
The fitted clockwise differences close exactly, while the residual distributes
the one-turn conflict among the four edges. A least-squares surface need not wrap
back to the measured phase. It minimizes gradient residuals over real values,
not integer pixel labels. With no residues and no aliased steps, integration can
fit the noisy phase itself exactly; least squares is not automatically a denoiser.
''','''
truth,phase,confidence=scene()
images([truth,phase,confidence],['Truth','Wrapped input','Pixel confidence'])
result,info=pv.unwrap(phase,return_info=True)
print(info); report(result,truth,phase)
''','''
# Parvaneh SQUARES pixel weights, then takes the minimum at each edge.
pixel_weight=np.array([1.,1.,.2,1.])
ends=[(0,1),(1,3),(2,3),(0,2)]
q=np.array([min(pixel_weight[a]**2,pixel_weight[b]**2) for a,b in ends])
weighted=np.linalg.lstsq(np.vstack([np.sqrt(q)[:,None]*D,np.ones(4)]),
                        np.r_[np.sqrt(q)*g,0],rcond=None)[0]
actual=pv.unwrap(w.reshape(2,2),pixel_weight.reshape(2,2),tol=1e-10)
assert np.allclose(actual.ravel(),weighted,atol=1e-7)
fig,ax=plt.subplots(figsize=(7,3))
ax.plot(D@u-g,'o-',label='Uniform'); ax.plot(D@weighted-g,'s-',label='Weighted')
ax.set(xticks=range(4),xticklabels=['AB','BC','DC','AD'],ylabel='Edge residual (rad)')
ax.legend(); plt.show()
weighted_large,wi=pv.unwrap(phase,confidence,return_info=True)
print(wi); print('Weighted aligned RMSE:',pv.rmse_aligned(weighted_large,truth))
''',r'''
For this API $q_{ab}=\min(c_a^2,c_b^2)$, not simply $c_a$.
The weighted equation is $D^TQD u=D^TQg$. Low-cost edges can absorb more
inconsistency, but low cost does not identify the correct physical jump.
Positive weights keep this small grid connected; zero weights may create
independent components with independent offsets. Replace nonfinite input with
finite placeholders before supplying zero weights.

`unwrap` supports any rank with every axis at least length two. DCT diagonalizes
the unweighted rectangular-grid operator; weighted problems use it as a
preconditioner in conjugate gradients. Accelerated backends are implementations
of the same method, not separate mathematical algorithms.
''','core')

lesson('quality_guided_unwrapping.ipynb','Quality-guided paths: watch the frontier grow',
       'Which already-reached neighbor should propose the next pixel’s phase?',r'''
On an accepted parent $a$ and candidate $b$, propose
$u_b=u_a+W(w_b-w_a)$. Rank candidates by their quality; accept the best queued
candidate, then propose its still-unseen neighbors. The accepted phase is frozen.

1. Seed a valid region. In this implementation the first seed is the first
   unvisited valid pixel in array order, not the globally best-quality pixel.
2. Queue neighbors with their parent and quality.
3. Accept the highest-priority queued candidate.
4. Continue; restart if another disconnected region remains.

For $u_a=2.8$ and $w_b=-2.9$, the raw difference is $-5.7$ but its principal
step is about $0.583$, so the proposed $u_b$ is about $3.383$ radians.
''','''
parent=2.8; child=-2.9; step=float(W(child-parent))
table(['Quantity','Radians'],[['Raw difference',child-parent],['Wrapped difference',step],['Proposal',parent+step]])
y,x=np.mgrid[0:5,0:6]; tiny=W(.8*x+.5*y)
quality=np.ones_like(tiny); quality[1:4,2]=.05
tiny_u,order=pv.quality_guided_unwrap(tiny,quality,return_order=True)
fig,ax=plt.subplots(figsize=(6,4)); im=ax.imshow(order,cmap='viridis')
for r,c in np.ndindex(order.shape): ax.text(c,r,str(order[r,c]),ha='center',va='center',color='white')
ax.set_title('Acceptance order: a low-quality column delays entry'); plt.colorbar(im,ax=ax); plt.show()
assert np.allclose(W(tiny_u-tiny),0)
''',r'''
The low-quality strip remains valid; it is delayed rather than removed. A poor
early parent decision can affect descendants. This is a frontier algorithm,
not the global edge sorting used by `reliability_unwrap`. The default quality
map and the string mode `quality='min_gradient'` also use different priorities.
''','''
truth,phase,confidence=scene()
u,order=pv.quality_guided_unwrap(phase,confidence,return_order=True)
images([phase,confidence,order],['Wrapped input','Quality','Acceptance order'])
report(u,truth,phase)
''','''
scores=[]
for label,q in [('Supplied',confidence),('Default',None),('Min gradient','min_gradient')]:
    u=pv.quality_guided_unwrap(phase,q)
    scores.append([label,f'{pv.rmse_aligned(u,truth):.4f}',f'{np.max(np.abs(W(u-phase))):.4f}'])
table(['Priority','Aligned RMSE','Raw wrapping error'],scores)
''',r'''
Only 2-D input is supported. `return_order` exposes traversal, not uncertainty.
The bounded frontier can postpone candidates; ties and `max_frontier` affect
order. Masked pixels are NaN. The legacy `min_gradient` path uses float32 cycle
coordinates and a shifted phase origin, visible in the raw wrapping error.
Mean-aligned RMSE removes a constant offset but does not certify absolute phase.
''','path_following')

for filename,title,fn,explanation in [
 ('goldstein_branch_cuts.ipynb','Goldstein branch cuts: neutralize a charged region','goldstein_unwrap',
  'Expand a search box around a charged cell. Gather other charges until the group is balanced, '
  'connecting them by cuts; if needed connect to the boundary. The cut map records forbidden traversal pixels. '
  'This is the method in which the integration path must not cross the cut. During path following, pixels on '
  'the cut are temporarily excluded, so a path cannot enter or cross the chain of cut pixels. This prevents '
  'two routes from enclosing a non-zero residue and disagreeing by a whole turn. Classical diagrams draw a '
  'branch cut as a curve joining residue cells; this implementation rasterizes that curve as a boolean pixel '
  'mask. It fills those pixels later from an already unwrapped neighbour.'),
 ('quality_mask_cuts.ipynb','Mask cuts: grow and thin a barrier','mask_cut_unwrap',
  'Start from charged cells, grow paths using the implementation’s minimum-gradient priority until charge '
  'is balanced or a boundary is reached, then thin the resulting mask while preserving required connections.')]:
    lesson(filename,title,'How can a path avoid enclosing an unbalanced residue?',r'''
For clockwise steps around a cell, $r=(g_{top}+g_{right}-g_{bottom}-g_{left})/(2\pi)$.
The charge is an integer because these steps come from wrapped pixel values.
A region containing charges $+1,-1$ is balanced: its total is zero. One containing
$+1,+1,-1$ still has charge $+1$ and cannot be treated as neutral.

'''+explanation+r'''

After constructing cuts, integrate only through allowed paths. Then fill cut
pixels from adjacent integrated pixels where possible. Cut placement is a
heuristic geometric decision, not the same optimization as minimum-cost flow.
''','''
rows=cols=11
y,x=np.mgrid[:rows,:cols]
vortex=(4.5,4.5); antivortex=(6.5,4.5)
pair_phase=W(np.angle((x-vortex[0])+1j*(y-vortex[1]))
             -np.angle((x-antivortex[0])+1j*(y-antivortex[1])))
charge=pv.phase_residues(pair_phase)
_,cut_pixels=pv.goldstein_unwrap(pair_phase,return_cuts=True,max_cut_length=20)
positive=np.argwhere(charge==1); negative=np.argwhere(charge==-1)
assert positive.tolist()==[[4,4]] and negative.tolist()==[[4,6]]
assert np.argwhere(cut_pixels).tolist()==[[4,5],[4,6]]
fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
axes[0].imshow(pair_phase,cmap='twilight',vmin=-np.pi,vmax=np.pi)
axes[0].set_title('Wrapped vortex pair')
axes[1].imshow(np.zeros_like(pair_phase),cmap='Greys',vmin=0,vmax=1)
axes[1].plot([4.5,6.5],[4.5,4.5],color='#d8a100',lw=6,
             label='conceptual branch cut')
axes[1].scatter([4.5,6.5],[4.5,4.5],c=['#b83d52','#2463a6'],s=220,zorder=3)
axes[1].text(4.5,4.5,'+1',color='white',ha='center',va='center',weight='bold')
axes[1].text(6.5,4.5,'−1',color='white',ha='center',va='center',weight='bold')
axes[1].set_title('Residues live in 2×2 cells')
axes[2].imshow(cut_pixels,cmap='Greys',vmin=0,vmax=1)
axes[2].scatter([4.5,6.5],[4.5,4.5],c=['#b83d52','#2463a6'],s=170,zorder=3)
axes[2].text(4.5,4.5,'+1',color='white',ha='center',va='center',weight='bold')
axes[2].text(6.5,4.5,'−1',color='white',ha='center',va='center',weight='bold')
axes[2].set_title('Returned boolean cut pixels')
for ax in axes:
    ax.set(xlabel='column',ylabel='row',xticks=range(0,11,2),yticks=range(0,11,2))
plt.show()
print('Positive residue [row, col]:',positive.tolist())
print('Negative residue [row, col]:',negative.tolist())
print('Cut pixels [row, col]:',np.argwhere(cut_pixels).tolist())
''',r'''
The middle panel shows the mathematical idea: the positive and negative residue
cells are paired by one branch cut. The right panel shows this implementation's
representation of that same connection. Black entries are actual pixels skipped
during the first integration pass, not edges between pixels. The coloured
markers remain at cell centres, which is why they are offset by half a sample
from the cut-pixel coordinates printed below. The next example shows the same
returned mask on the larger synthetic scene.
''',f'''
truth,phase,confidence=scene()
u,cuts=pv.{fn}(phase,return_cuts=True)
images([phase,pv.phase_residues(phase),cuts],['Wrapped phase','Cell residues','Computed cut pixels'])
print('Cut pixels:',np.count_nonzero(cuts)); report(u,truth,phase)
''','''
clean=.3*np.add.outer(np.arange(8),np.arange(9)); wrapped=W(clean)
rows=[]
for name,method in [('Goldstein',pv.goldstein_unwrap),('Mask cut',pv.mask_cut_unwrap)]:
    v,cut=method(wrapped,return_cuts=True)
    rows.append([name,int(cut.sum()),f'{pv.rmse_aligned(v,clean):.2e}',
                 f'{np.mean(v-clean):.4f}'])
table(['Method','Cuts on clean ramp','Aligned RMSE','Mean offset (rad)'],rows)
''',r'''
These are 2-D algorithms. Their float32 cycle representation shifts the phase
origin by approximately $\pi$ on a clean ramp; the printed mean offset reveals
that convention. Cut filling can leave zero values when no adjacent integrated
pixel exists, and excluded mask pixels are also zero. Inspect the cut map and
validity mask; a finite number alone does not guarantee a recovered measurement.
Neither short cuts nor few cut pixels guarantees lower error against truth.
''','goldstein')

lesson('flynn_minimum_discontinuity.ipynb','Flynn: reduce integer discontinuities',
       'Can shifting an entire region by one turn reduce the number of costly jumps?',r'''
Represent a candidate as $u=w+2\pi n$ (up to the implementation’s phase origin).
For an edge, the integer discrepancy from its shortest wrapped step is
$k_{ab}=\operatorname{round}((u_b-u_a-W(w_b-w_a))/(2\pi))$.
A minimum-discontinuity model penalizes $\sum c_{ab}|k_{ab}|$.

1. Initialize integer jumps from the wrapped input.
2. Sweep the padded network in four directions, propagating possible improvements.
3. Use detected improving paths/loops to change integer discontinuities.
4. Repeat until a sweep cannot add an improving edge, then reconstruct the phase.

On a chain $w=(2.8,-2.9,-2.3)$, shifting the last two pixels together by $2\pi$
removes the first jump without changing the difference inside that region.
''','''
w=np.array([2.8,-2.9,-2.3]); candidates=[w,w+2*np.pi*np.array([0,1,1])]
fig,ax=plt.subplots(figsize=(7,3))
for u,label in zip(candidates,['Initial labels','Shift region {1,2} by one turn']):
    ax.plot(u,'o-',label=label)
    k=np.rint((np.diff(u)-W(np.diff(w)))/(2*np.pi)).astype(int)
    print(label,'integer edge discrepancies:',k,'L1 total:',np.abs(k).sum())
ax.set(xlabel='pixel',ylabel='phase (rad)'); ax.legend(); plt.show()
''',r'''
Changing a whole connected region preserves its internal gradients and changes
only its boundary. The chain illustrates why region moves can improve an answer
when isolated pixel changes would introduce new jumps. The full Flynn routine
uses its own network sweeps, not the three-pixel enumeration above.
''','''
truth,phase,confidence=scene()
u,iterations=pv.flynn_unwrap(phase,quality=confidence,return_iterations=True)
images([phase,confidence],['Wrapped phase','Quality before normalization'])
print('Network sweeps:',iterations); report(u,truth,phase)
''','''
rows=[]
for name,q in [('Uniform',None),('Patch confidence',confidence)]:
    u,it=pv.flynn_unwrap(phase,quality=q,return_iterations=True)
    rows.append([name,it,f'{pv.rmse_aligned(u,truth):.4f}'])
table(['Quality','Sweeps','Aligned RMSE'],rows)
''',r'''
Quality is linearly rescaled to [0,1], so it is not an absolute inverse variance.
The implementation retains integer costs and float32 cycle arithmetic. As with
the branch-cut routines, the phase-origin offset is visible in the raw wrapping
metric; compare gradients and aligned errors as well. The API requires 2-D input.
An iteration count describes work performed, not a certificate of physical truth.
''','flynn')

lesson('robust_lp_unwrapping.ipynb','Robust Lp: reweight the residuals',
       'How can an objective stop a few large gradient errors from dominating the entire fit?',r'''
Let $e=Du-g$. Instead of $\sum e^2$, minimize the smoothed objective
$J_p=\sum(e^2+\epsilon^2)^{p/2}$ for $1\le p\le2$.
Differentiating a term gives $p e(e^2+\epsilon^2)^{p/2-1}$.
Freeze the factor $q=(e^2+\epsilon^2)^{p/2-1}$ and solve weighted least squares.

1. Start from ordinary least squares.
2. Compute residuals on horizontal and vertical edges.
3. Update $q$; large residuals receive less relative weight when $p<2$.
4. Solve for a new surface with those edge weights.
5. Stop on relative change or the outer iteration limit.

For $p=1$ and small $\epsilon$, residuals 0.1 and 2 have weights near 10 and 0.5.
This is adaptive residual weighting, not supplied measurement confidence.
''','''
e=np.linspace(-3,3,301); eps=.05
fig,axes=plt.subplots(1,2,figsize=(9,3))
for p in [1.,1.2,2.]:
    axes[0].plot(e,(e*e+eps*eps)**(p/2),label=f'p={p}')
    axes[1].plot(e,(e*e+eps*eps)**(p/2-1),label=f'p={p}')
axes[0].set(title='Smoothed penalty',xlabel='residual',ylabel='cost')
axes[1].set(title='IRLS weight',xlabel='residual',ylabel='weight'); axes[1].set_yscale('log')
for ax in axes: ax.legend()
plt.show()
''',r'''
Reducing $p$ makes a large residual less expensive relative to many smaller
residuals. The algorithm may concentrate disagreement instead of spreading it.
The positive $\epsilon$ prevents an infinite weight at zero residual, but also
changes the objective near zero. A robust fit still need not preserve wrapped values.
''','''
truth,phase,confidence=scene()
u,info=pv.unwrap_lp(phase,p=1.2,outer_iter=12,return_info=True)
print(info); images([truth,phase],['Truth','Wrapped input']); report(u,truth,phase)
''','''
rows=[]
for p in [2.,1.5,1.2,1.]:
    u,info=pv.unwrap_lp(phase,p=p,return_info=True)
    rows.append([p,info.outer_iterations,f'{info.objective:.3f}',f'{pv.rmse_aligned(u,truth):.4f}'])
table(['p','Outer iterations','Own objective','Aligned RMSE'],rows)
assert np.allclose(pv.unwrap_lp(phase,p=2),pv.unwrap(phase))
''',r'''
Objective numbers for different $p$ measure different functions and cannot be
ranked as if they were the same score. Compare a shared metric such as aligned
RMSE instead. The current `unwrap_lp` API requires 2-D data and has no external
weight argument. A smaller $p$ is not guaranteed to improve a particular scene.
''','minimum_norm')

lesson('multigrid_unwrapping.ipynb','Multigrid: correct errors at the scale where they are visible',
       'Why can a coarse grid accelerate the solution of a fine-grid phase problem?',r'''
For $Au=b$ with $A=D^TQD$, define residual $r=b-Au$. If the error is
$e=u_* -u$, then $Ae=r$. This is the correction equation.

1. Relax on the fine grid to reduce rapidly varying error.
2. Restrict $r$ to a coarser grid.
3. Approximately solve the coarse error equation.
4. Prolong the correction and add it to the fine estimate.
5. Relax again; repeat these V-cycles until the relative residual is small.

A slowly varying error needs many local sweeps on a fine grid. On a coarse grid
the same shape spans fewer nodes, so local relaxation can address it more quickly.
''','''
x=np.linspace(0,1,65); smooth=np.sin(np.pi*x); rapid=.25*np.sin(16*np.pi*x)
fig,axes=plt.subplots(1,2,figsize=(9,3))
axes[0].plot(x,smooth+rapid,label='Combined error'); axes[0].plot(x,smooth,'--',label='Slow component')
axes[1].plot(x,smooth,color='gray'); axes[1].plot(x[::8],smooth[::8],'o-',label='Coarse samples')
axes[0].set_title('Different error scales'); axes[1].set_title('Slow error on a coarse grid')
for ax in axes: ax.set(xlabel='position',ylabel='illustrative error'); ax.legend()
plt.show()
''',r'''
This is an illustration of error scales, not a record of solver iterations.
Next we measure actual solves with different cycle budgets. Multigrid changes
how the linear equations are solved; it does not change least squares into an
integer-label method or remove its offset ambiguity.
''','''
y,x=np.mgrid[0:25,0:29]; truth=.2*x+.3*y+np.sin(x/5)
phase=W(truth)
u,info=pv.multigrid_unwrap(phase,max_cycles=80,tol=1e-8,return_info=True)
print(info); report(u,truth,phase)
''','''
budgets=[1,2,4,8,16,32]; reference=pv.unwrap(phase)
fig,ax=plt.subplots(figsize=(7,3))
for levels,label in [(1,'One grid'),(None,'Grid hierarchy')]:
    errors=[]
    for budget in budgets:
        value=pv.multigrid_unwrap(phase,levels=levels,max_cycles=budget)
        errors.append(max(pv.rmse_aligned(value,reference),1e-15))
    ax.semilogy(budgets,errors,'o-',label=label)
ax.set(xlabel='Maximum V-cycles (fresh solve)',ylabel='RMSE against direct LS (rad)'); ax.legend(); plt.show()
''',r'''
These points are fresh solves with increasing budgets, not timing measurements.
Inspect the convergence report: reaching `max_cycles` is not convergence.
Strong weight contrasts may stall or destabilize the hierarchy. The method
supports N-D arrays, but coarse levels stop when axes become too short.
Masked samples do not automatically become NaN; callers must reapply their mask.
''','multigrid')

lesson('graph_cut_unwrapping.ipynb','Graph cuts: move many integer labels together',
       'How can a binary cut decide which pixels should gain a whole turn?',r'''
Write $u_p=w_p+2\pi K_p$ with integer labels $K_p$. The implemented objective is
$E(K)=\sum_{ab}c_{ab}|w_b-w_a+2\pi(K_b-K_a)|^p$.
This penalizes **corrected phase steps**, not least-squares residuals $Du-g$.

For a move $K'=K+s x$, each $x_p$ is 0 or 1. An edge has four possible costs:
$V(d),V(d+2\pi s),V(d-2\pi s),V(d)$ for $(x_a,x_b)=(0,0),(0,1),(1,0),(1,1)$.
Convex $V$ gives a submodular binary problem represented by a cut.

1. Build the current pairwise label energy.
2. Find the best simultaneous binary move for a scheduled turn step.
3. Accept energy-reducing moves; repeat steps and sweeps.
4. Return integer labels as a phase field and report termination.
''','''
d=-5.7; span=2*np.pi; states=[(0,0),(0,1),(1,0),(1,1)]
costs=[abs(d+span*(b-a)) for a,b in states]
table(['Move (a,b)','New difference','Cost p=1'],
      [(str((a,b)),f'{d+span*(b-a):.3f}',f'{v:.3f}') for (a,b),v in zip(states,costs)])
assert costs[0]+costs[3]<=costs[1]+costs[2]+1e-12
fig,ax=plt.subplots(figsize=(7,3)); ax.bar([str(s) for s in states],costs,color='#2463a6')
ax.set(xlabel='Binary move',ylabel='Pairwise energy',title='One edge in a graph-cut move'); plt.show()
''',r'''
Moving both endpoints changes no difference, while moving only the second
endpoint can remove a wrap jump. The cut chooses many endpoints together.
Do not equate this energy with $\sum c|k|$: generally
$|g+2\pi k|\ne2\pi|k|$. Likewise $p=2$ here is not the real-valued
least-squares gradient-fitting objective.
''','''
truth,phase,confidence=scene()
u,info=pv.puma_unwrap(phase,weight=confidence,p=1,max_jump=2,return_info=True)
print(info); images([phase,np.rint((u-phase)/(2*np.pi))],['Wrapped phase','Integer pixel labels'])
report(u,truth,phase)
''','''
rows=[]
for p in [1.,2.]:
    value,info=pv.puma_unwrap(phase,p=p,max_jump=2,return_info=True)
    rows.append([p,f'{pv.rmse_aligned(value,truth):.4f}'])
    print(info)
table(['p','Aligned RMSE'],rows)
v=W(.4*np.indices((3,4,5)).sum(axis=0))
out=pv.puma_unwrap(v); assert out.shape==v.shape and np.allclose(W(out-v),0)
print('Volume shape:',out.shape)
''',r'''
The API accepts at least two axes, including volumes. `max_jump` bounds a move
step, not the total number of turns in a label. A finite sweep budget is a
termination condition, not by itself a global-optimality certificate. Zero
confidence removes samples and returns NaN there. The default gauge keeps a
reference pixel; other gauges can alter the absolute wrapped origin.
''','graph_cut')

lesson('nd_cycle_flow.ipynb','Volume flow: close loops in every plane',
       'What changes when the grid has a third neighbor direction?',r'''
For each oriented face $C$, compute residue $r_C$ from wrapped edge steps.
Integer corrections satisfy $Bk=-r$, where $B$ contains the face-edge signs.
Minimize a correction cost subject to these constraints.

In 2-D a pixel edge has at most two adjacent cells, giving the familiar network
structure. A 3-D interior edge can belong to four faces. A planar flow solver
cannot simply treat those four faces as its two arc endpoints.

1. Enumerate grid edges and faces in all coordinate planes.
2. Build $Bk=-r$ and a correction objective.
3. Use planar flow in 2-D or the integer search for a small volume.
4. Integrate a feasible correction and inspect `info.proven`.

A $2\times2\times2$ cube has 8 vertices, 12 edges, and 6 square faces.
There are only $12-8+1=5$ independent cycles; one face equation is redundant.
''','''
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(5,4)); ax=fig.add_subplot(111,projection='3d')
nodes=np.array(list(np.ndindex(2,2,2)))
for a in nodes:
    ax.scatter(*a,color='#2463a6')
    for axis in range(3):
        b=a.copy(); b[axis]+=1
        if b[axis]<2: ax.plot(*np.array([a,b]).T,color='#2463a6')
ax.set(title='Six faces constrain twelve cube edges',xlabel='axis 0',ylabel='axis 1',zlabel='axis 2'); plt.show()
''',r'''
For a rectangular $n_0\times n_1\times n_2$ grid the number of faces is
$(n_0-1)(n_1-1)n_2+(n_0-1)n_1(n_2-1)+n_0(n_1-1)(n_2-1)$.
Counting faces is not the same as counting independent cycles. Redundant
constraints are mathematically harmless but increase the work of a generic solve.
''','''
z,y,x=np.indices((3,4,5)); truth=2.2*z+.3*y+.4*x; phase=W(truth)
u,info=pv.flow_nd_unwrap(phase,method='ilp',max_nodes=200,return_info=True)
print(info); assert u.shape==truth.shape
images([phase[1],align(u,truth)[1],(align(u,truth)-truth)[1]],['Middle wrapped slice','Joint solution','Joint error'])
print('Joint aligned RMSE:',pv.rmse_aligned(u,truth))
''','''
independent=np.stack([pv.network_flow_unwrap(frame) for frame in phase])
slice_mode,si=pv.flow_nd_unwrap(phase,method='slice',return_info=True)
fig,ax=plt.subplots(figsize=(7,3))
for a,label in [(u,'Joint ILP'),(independent,'Independent 2-D'),(slice_mode,'API slice baseline')]:
    ax.plot(np.mean(align(a,truth)-truth,axis=(1,2)),'o-',label=label)
ax.set(xlabel='slice',ylabel='Mean error after ONE global alignment'); ax.legend(); plt.show()
print(si)
''',r'''
Per-slice alignment would hide the relative-offset failure this experiment is
designed to expose. The small smooth volume keeps exact search inexpensive;
noisy larger volumes can require many relaxations. `max_nodes` can return an
unproven incumbent or raise if no feasible solution is found. Inspect the report.
The public slice mode is specifically a 3-D baseline. Dimension-independent
least squares and reliability sorting remain separate options for large volumes.
''','flow_nd')

lesson('statistical_cost_unwrapping.ipynb','Statistical costs: turn uncertainty into a price',
       'When should a phase jump be explained as noise, and when should the model tolerate a discontinuity?',r'''
A statistical cost assigns a price to each candidate integer correction.
In a quadratic prior model, write $C_e(k)=a_e(k-\mu_e)^2$, where $\mu_e$
is a predicted correction in turns and $a_e$ is inverse-variance-like curvature.
For $\mu=0.2$, candidate $k=0$ has residual $-0.2$ turns; $k=1$ has $0.8$.
At equal curvature, their costs are in the ratio $0.04:0.64$.

1. Estimate directional mean steps and uncertainty from supplied coherence.
2. Convert them into edge correction costs.
3. Solve the convex model under cycle constraints.
4. In deformation mode, optionally refine a nonconvex shelf model that makes
   some large discrepancies less costly; inspect its proof flag.
''','''
k=np.arange(-3,4); mu=.2
fig,ax=plt.subplots(figsize=(7,3))
for variance in [.1,1.]: ax.plot(k,(k-mu)**2/variance,'o-',label=f'Illustrative variance={variance}')
ax.set(xlabel='Candidate integer turns',ylabel='Quadratic cost',title='Uncertainty changes the penalty'); ax.legend(); plt.show()
table(['k','Residual in turns','Cost at variance 1'],[(i,f'{i-mu:.1f}',f'{(i-mu)**2:.2f}') for i in [0,1]])
''',r'''
These curves illustrate a quadratic prior; the actual implementation estimates
its variance and mean through the documented sensor and box-average model.
Coherence is supplied data, not the same thing as the phase-only reliability
score. Calling a phase score coherence does not make it a calibrated measurement.
''','''
truth,phase,confidence=scene()
coherence=np.where(confidence<1,.35,.9)
params=pv.StatCostParams(kperpdpsi=3,kpardpsi=3)
u,info=pv.stat_cost_unwrap(phase,coherence,params=params,costmode='smooth',return_info=True)
print(info); images([phase,coherence],['Wrapped phase','Synthetic supplied coherence']); report(u,truth,phase)
''','''
rows=[]
for mode,shelf in [('smooth',False),('defo',False),('defo',True)]:
    value,info=pv.stat_cost_unwrap(phase,coherence,params=params,costmode=mode,shelf=shelf,return_info=True)
    rows.append([mode,shelf,info.proven,f'{pv.rmse_aligned(value,truth):.4f}'])
table(['Model','Shelf requested','Proven','Aligned RMSE'],rows)
''',r'''
The smooth convex solve and capped deformation refinement have different
objectives. A proof for the former does not transfer to the latter. Window sizes
must fit the directional gradient arrays; this small scene explicitly uses a
3-pixel window. This 2-D implementation is inspired by statistical network
models; it is not a call to an external SNAPHU executable and should not be
presented as a complete drop-in numerical reproduction of it.
''','stat_costs')

lesson('space_time_priors.ipynb','Space-time: predict an edge from its history',
       'How can neighboring dates help price the integer decision in one image?',r'''
For an edge observed at times $t_j$, Gaussian weights around $t_i$ are
$a_{ij}\propto\exp(-(t_i-t_j)^2/(2\tau^2))$. Normalize them to sum to one.
Average phases on the unit circle: $m_i=\sum_j a_{ij}e^{ig_j}$.
Its angle is a local reference; fit wrapped deviations around that reference
with a weighted line to estimate a prediction and uncertainty.

1. Extract spatial wrapped steps for every date.
2. Build temporal predictions and error variances for each spatial edge.
3. Price integer corrections using these priors.
4. Solve each date's spatial network and integrate it.

This uses a stack but does not add temporal edges to the spatial flow graph.
Each output image still has its own phase-offset ambiguity.
''','''
days=np.arange(6)*12.; tau=24.; angles=np.array([2.8,3.,-3.1,-2.9,-2.7,-2.5])
weights=np.exp(-(days-days[2])**2/(2*tau*tau)); weights/=weights.sum()
mean=np.angle(np.sum(weights*np.exp(1j*angles)))
fig,axes=plt.subplots(1,2,figsize=(9,3))
axes[0].bar(days,weights,width=8); axes[0].set(title='Normalized temporal weights',xlabel='day')
axes[1].plot(days,angles,'o-',label='Wrapped angles'); axes[1].axhline(mean,color='red',label='Circular mean')
axes[1].set(xlabel='day',ylabel='rad'); axes[1].legend(); plt.show()
print('Arithmetic mean:',angles.mean(),'circular mean:',mean)
''',r'''
The arithmetic mean can fall near zero even when measurements cluster near
$\pi$. A circular mean avoids that branch-cut artifact. The actual temporal
prior additionally fits a local line and estimates variance; the two-line mean
calculation is just the first stage, not a substitute implementation.
''','''
days=np.arange(6)*12.; y,x=np.mgrid[0:8,0:10]
truth=np.array([(.3+.02*i)*x+.2*y+.5*i for i in range(6)])
phase=W(truth+np.random.default_rng(8).normal(0,.08,truth.shape))
params=pv.SpaceTimeParams(time_win=24)
u,info=pv.space_time_unwrap(phase,days,params=params,return_info=True)
print(info)
images([phase[3],align(u[3],truth[3]),align(u[3],truth[3])-truth[3]],['Wrapped date 3','Aligned reconstruction','Error'])
''','''
rows=[]
for window in [12.,48.]:
    value=pv.space_time_unwrap(phase,days,params=pv.SpaceTimeParams(time_win=window))
    errors=[pv.rmse_aligned(a,b) for a,b in zip(value,truth)]
    rows.append([window,f'{np.mean(errors):.4f}'])
table(['Window (days)','Mean per-image aligned RMSE'],rows)
fig,ax=plt.subplots(figsize=(7,3))
ax.plot(days,truth[:,3,4]-truth[:,0,0],'o-',label='True relative phase')
ax.plot(days,u[:,3,4]-u[:,0,0],'s--',label='Recovered relative phase')
ax.set(xlabel='day',ylabel='Phase relative to pixel (0,0)'); ax.legend(); plt.show()
''',r'''
The input shape is (observations, rows, columns), and acquisition days must
increase. Longer smoothing windows assume slower temporal change; real rapid
deformation may violate that assumption. Time does not fix per-image constants,
so the plot uses a spatial reference and the metric aligns each frame.
The alternative `design` input represents interferograms through an epoch design
matrix; it is a different input contract, not a fourth spatial axis.
''','space_time')

lesson('extended_minimum_cost_flow.ipynb','EMCF: connect acquisitions before unwrapping space',
       'What extra consistency is supplied by a triangle of acquisition pairs?',r'''
For acquisition phases $\phi_0,\phi_1,\phi_2$, pair differences obey
$(\phi_1-\phi_0)+(\phi_2-\phi_1)-(\phi_2-\phi_0)=0$.
Wrapped measurements can differ from this relation by whole turns.

EMCF uses temporal loops of **spatial edge measurements**. For temporal loop
signs $s_{cj}$, corrections satisfy $\sum_j s_{cj}k_j=-r_c$.

1. Specify acquisition pairs as links (earlier, later).
2. Form interferograms, or supply them directly in link order.
3. Correct temporal loop discrepancies for every spatial edge.
4. Run the spatial correction stage for each interferogram and integrate.

The two stages enforce their own constraints. Do not infer that the final
spatial stage automatically certifies every possible joint space-time objective.
''','''
links=np.array([[0,1],[1,2],[0,2]])
history=np.array([0.,2.5,5.])
pair=W(history[links[:,1]]-history[links[:,0]])
print('Wrapped pair measurements:',pair)
print('Triangle circulation in turns:',(pair[0]+pair[1]-pair[2])/(2*np.pi))
fig,ax=plt.subplots(figsize=(6,4)); pos=np.array([[0,0],[1,1],[2,0]])
for j,(a,b) in enumerate(links):
    ax.annotate('',xy=pos[b],xytext=pos[a],arrowprops=dict(arrowstyle='->',lw=2))
    middle=(pos[a]+pos[b])/2; ax.text(*middle,f'link {j}',color='#2463a6',bbox=dict(facecolor='white',edgecolor='none'))
for i,p in enumerate(pos): ax.scatter(*p,s=400,color='#dbe8f5'); ax.text(*p,str(i),ha='center',va='center')
ax.set(xlim=(-.3,2.3),ylim=(-.3,1.3),title='Acquisition graph: one temporal cycle'); ax.axis('off'); plt.show()
''',r'''
The triangle is acquisition geometry, not three neighboring image pixels.
For the full algorithm each spatial edge supplies its own vector across these
links. `emcf_unwrap` takes acquisition phases; `emcf_unwrap_interferograms`
takes already-formed pair measurements. Both return one image per link.
''','''
y,x=np.mgrid[0:6,0:7]
acquisition_truth=np.array([(.3+.12*i)*x+.2*y+.6*i for i in range(3)])
acquisitions=W(acquisition_truth)
pair_truth=acquisition_truth[links[:,1]]-acquisition_truth[links[:,0]]
pair_phase=W(acquisitions[links[:,1]]-acquisitions[links[:,0]])
u,info=pv.emcf_unwrap(acquisitions,links=links,return_info=True)
print(info); assert u.shape==pair_truth.shape
images([pair_phase[2],align(u[2],pair_truth[2]),align(u[2],pair_truth[2])-pair_truth[2]],
       ['Wrapped pair 0→2','Reconstruction','Error'])
''','''
v,info=pv.emcf_unwrap_interferograms(pair_phase,links,return_info=True)
assert np.allclose(u,v)
table(['Link','Aligned RMSE'],[(str(tuple(link)),f'{pv.rmse_aligned(a,b):.3e}')
                            for link,a,b in zip(links,u,pair_truth)])
print('Both input APIs agree on the same pair measurements.')
fig,ax=plt.subplots(figsize=(7,3))
closure=u[0]+u[1]-u[2]; ax.plot(closure.ravel(),'o',ms=3)
ax.set(xlabel='flattened pixel',ylabel='Final triangle closure (rad)',title='Measure closure; do not assume it'); plt.show()
''',r'''
Default links require at least four acquisitions; the three-epoch example
therefore supplies explicit links. Pair order determines signs. A mask and
weight have the shape of one spatial image, not the acquisition stack.
This clean example validates the two API routes but is not a noisy-scene benchmark.
Individual interferograms retain separate offset conventions; measure temporal
closure after processing if your downstream analysis requires it.
''','emcf')

# Split the already-developed visual walkthrough into two self-contained lessons.
source=nb.read(ROOT/'notebooks'/'reliability_and_flow_notes.ipynb',as_version=4)
def extract(filename,title,indices):
    import copy
    selected=[nb.v4.new_markdown_cell(title)]+[copy.deepcopy(source.cells[i]) for i in indices]
    for c in selected:
        if c.cell_type=='code': c.outputs=[]; c.execution_count=None
    result=nb.v4.new_notebook(cells=selected,metadata=source.metadata)
    nb.validate(result); nb.write(result,ROOT/'notebooks'/filename)

# Content-based section boundaries keep extraction independent of cell counts.
starts={}
for i,c in enumerate(source.cells):
    if c.cell_type=='markdown' and c.source.startswith('## '):
        starts[c.source.split(' · ')[0]]=i
start_flow=starts['## 4']; start_scene=starts['## 6']
extract('reliability_sorting.ipynb','# Reliability sorting · from four pixels to a spanning tree',
        list(range(1,start_flow)))
extract('minimum_cost_flow.ipynb','# Minimum-cost flow · repair a residue network',
        [1,2,3]+list(range(start_flow,start_scene)))
print('Built 14 method-family notebooks (including the expanded least-squares lesson).')
