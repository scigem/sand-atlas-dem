"""Bounded symmetric3x3 matrix exponential in Mandel coordinates.

Version symmetric-expm-taylor18-frobenius64-v1. Fixed-degree Horner polynomial,
Cartesian products and scaling/squaring; no eigenvalue-gap branch or clipping.
"""
import math
VERSION='symmetric-expm-taylor18-frobenius64-v1'
MAX_FROBENIUS=64.

def symmetric_expm(values):
 if not isinstance(values,(list,tuple))or len(values)!=6 or any(type(x)not in(int,float)for x in values):raise ValueError('expm requires six finite real Mandel components')
 values=[float(x)for x in values]
 if not all(math.isfinite(x)for x in values):raise ValueError('expm requires finite components')
 frobenius=0.
 for x in values:frobenius=math.hypot(frobenius,x)
 if frobenius>MAX_FROBENIUS:raise ValueError('expm unsupported Frobenius norm above64')
 root2=math.sqrt(2.);x,y,z,u,v,w=values
 a=[[x,u/root2,w/root2],[u/root2,y,v/root2],[w/root2,v/root2,z]]
 row_norm=max(sum(abs(x)for x in row)for row in a);s=0
 while row_norm>.5:row_norm*=.5;s+=1
 b=[[math.ldexp(x,-s)for x in row]for row in a]
 def checked(x):
  if not math.isfinite(x):raise ValueError('nonfinite expm intermediate')
  return x
 def multiply(a,b):
  out=[[0.]*3 for _ in range(3)]
  for i in range(3):
   for j in range(3):
    for k in range(3):out[i][j]=checked(out[i][j]+checked(a[i][k]*b[k][j]))
  return out
 def symmetric(a):
  for i in range(3):
   for j in range(i):a[i][j]=a[j][i]=checked(.5*a[i][j]+.5*a[j][i])
  return a
 p=[[float(i==j)for j in range(3)]for i in range(3)]
 for k in range(18,0,-1):
  product=multiply(b,p)
  p=symmetric([[checked(float(i==j)+product[i][j]/k)for j in range(3)]for i in range(3)])
 for _ in range(s):p=symmetric(multiply(p,p))
 return [p[0][0],p[1][1],p[2][2],checked(root2*p[0][1]),checked(root2*p[1][2]),checked(root2*p[0][2])]
