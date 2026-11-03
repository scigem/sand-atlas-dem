"""Continuous isotropic reference for the declared Atlas Cam-Clay equations.

This is an analytical reduction, not a second implementation of the finite-step
residual and not a source/experiment validation. Strains are compression positive.
"""
import math


def cam_clay_isotropic(pressure, pc, e, kappa, lambda_c, volumetric_strains):
    """Return p, pc, e at ordered absolute volumetric strains starting at zero.

    Each consecutive pair describes monotone isotropic loading/unloading. The
    sequence may reverse and repeat. On the yield surface p=pc; elastic unloading
    retains pc. Specific volume v=1+e follows v=v0*exp(-epsilon_v).
    """
    values=[pressure,pc,e,kappa,lambda_c,*volumetric_strains]
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
        raise ValueError('finite scalar reference inputs required')
    if not (0<pressure<=pc and e>0 and 0<kappa<lambda_c):
        raise ValueError('reference requires 0<p<=pc, e>0 and 0<kappa<lambda_c')
    if not volumetric_strains or volumetric_strains[0]!=0:
        raise ValueError('reference strain sequence must start at zero')
    v0=1+e;previous_v=v0;points=[]
    for strain in volumetric_strains:
        v=v0*math.exp(-strain)
        if not math.isfinite(v) or v<=1:
            raise ValueError('reference leaves positive-void-ratio domain')
        compression=previous_v-v
        elastic_log=math.log(pressure)+compression/kappa
        if compression>0 and elastic_log>math.log(pc):
            # Exact elastic entry consumes kappa*log(pc/p) of volume change.
            plastic_compression=compression-kappa*math.log(pc/pressure)
            pressure=pc*math.exp(plastic_compression/lambda_c);pc=pressure
        else:
            pressure*=math.exp(compression/kappa)
        if not math.isfinite(pressure) or pressure<=0:
            raise ValueError('reference pressure outside finite positive range')
        points.append(dict(volumetric_strain=strain,p=pressure,pc=pc,e=v-1))
        previous_v=v
    return points
