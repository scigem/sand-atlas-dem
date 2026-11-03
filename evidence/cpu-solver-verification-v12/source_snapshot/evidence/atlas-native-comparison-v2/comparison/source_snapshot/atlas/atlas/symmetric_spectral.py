"""Generic binary64 symmetric spectrum; Python mirror of frozen primitive v1.

Same 24-sweep/scaling/isolation policy. Independent accuracy references live in
catalogue/investigations/symmetric-spectral-primitive-v1, not this algorithm.
"""
import math
import sys
VERSION='symmetric3-scaled-cyclic-jacobi24-v1-candidate'
EPS=sys.float_info.epsilon
ROOT2=math.sqrt(2.)
SPECTRAL_FUNCTIONS={
 'principal_max_value':('max_value','scalar',True),
 'principal_mid_value':('mid_value','scalar',True),
 'principal_min_value':('min_value','scalar',True),
 'principal_max':('max','scalar',False),
 'principal_mid':('mid','scalar',False),
 'principal_min':('min','scalar',False),
 'principal_upper_sum':('upper_sum','scalar',False),
 'principal_lower_sum':('lower_sum','scalar',False),
 'projector_max':('projector_max','sym',False),
 'projector_min':('projector_min','sym',False),
}
def _norm(a):
 n=0.
 for row in a:
  for x in row:n=math.hypot(n,x)
 return n

def spectrum(values):
 if not isinstance(values,(tuple,list)) or len(values)!=6 or any(type(v)not in(int,float)for v in values):raise ValueError('nonfinite_input')
 try:values=[float(v)for v in values]
 except (ValueError,OverflowError)as exc:raise ValueError('nonfinite_input')from exc
 if not all(map(math.isfinite,values)):raise ValueError('nonfinite_input')
 largest=max(map(abs,values));exponent=math.frexp(largest)[1]if largest else 0
 diagonal=all(v==0 for v in values[3:]);s=[math.ldexp(v,-exponent)for v in values]
 if not diagonal and any(x!=0 and y==0 for x,y in zip(values,s)):raise ValueError('scaling_underflow')
 original=[[s[0],s[3]/ROOT2,s[5]/ROOT2],[s[3]/ROOT2,s[1],s[4]/ROOT2],[s[5]/ROOT2,s[4]/ROOT2,s[2]]]
 if not diagonal and any(v!=0 and v/ROOT2==0 for v in s[3:]):raise ValueError('shear_decode_underflow')
 vectors=[[float(i==j)for j in range(3)]for i in range(3)]
 raw=None
 if diagonal:
  order=sorted(range(3),key=lambda i:values[i],reverse=True);raw=[values[i]for i in order];eigen=[s[i]for i in order];vectors=[[float(i==order[j])for j in range(3)]for i in range(3)]
 else:
  a=[row[:]for row in original]
  for sweep in range(24):
   for p,q in ((0,1),(0,2),(1,2)):
    off=a[p][q]
    if off==0:continue
    delta=.5*a[q][q]-.5*a[p][p];denom=delta+math.copysign(math.hypot(delta,off),delta)
    if denom==0 or not math.isfinite(denom):raise ValueError('rotation_range')
    t=off/denom;c=1/math.hypot(1.,t);sn=t*c
    a[p][p]-=t*off;a[q][q]+=t*off;a[p][q]=a[q][p]=0.
    for k in range(3):
     if k!=p and k!=q:
      kp,kq=a[k][p],a[k][q];a[k][p]=a[p][k]=c*kp-sn*kq;a[k][q]=a[q][k]=sn*kp+c*kq
     vp,vq=vectors[k][p],vectors[k][q];vectors[k][p]=c*vp-sn*vq;vectors[k][q]=sn*vp+c*vq
  order=sorted(range(3),key=lambda i:a[i][i],reverse=True);eigen=[a[i][i]for i in order];vectors=[[row[j]for j in order]for row in vectors]
 residual=orthogonality=0.
 for i in range(3):
  for j in range(3):
   av=vv=0.
   for k in range(3):av+=original[i][k]*vectors[k][j];vv+=vectors[k][i]*vectors[k][j]
   residual=math.hypot(residual,av-vectors[i][j]*eigen[j]);orthogonality=math.hypot(orthogonality,vv-float(i==j))
 norm=_norm(original)
 if not math.isfinite(residual)or not math.isfinite(orthogonality)or residual>128*EPS*norm or orthogonality>128*EPS:raise ValueError('decomposition_not_converged')
 return eigen,vectors,exponent,raw,norm

def spectral(values,operation):
 eigen,vectors,exponent,raw,norm=spectrum(values)
 selected=2 if operation in ('min_value','min','upper_sum','projector_min')else 1 if operation in ('mid_value','mid')else 0
 if not operation.endswith('_value'):
  if any(not abs(eigen[selected]-eigen[j])>1024*EPS*norm for j in range(3)if j!=selected):raise ValueError('selected_eigenvalue_unresolved')
 if operation.startswith('projector_'):
  v=[row[selected]for row in vectors]
  return (v[0]*v[0],v[1]*v[1],v[2]*v[2],ROOT2*v[0]*v[1],ROOT2*v[1]*v[2],ROOT2*v[0]*v[2])
 source=raw if raw is not None else eigen
 y=source[0]+source[1]if operation=='upper_sum'else source[1]+source[2]if operation=='lower_sum'else source[selected]
 try:y=y if raw is not None else math.ldexp(y,exponent)
 except OverflowError as exc:raise ValueError('output_range')from exc
 if not math.isfinite(y):raise ValueError('output_range')
 return y
