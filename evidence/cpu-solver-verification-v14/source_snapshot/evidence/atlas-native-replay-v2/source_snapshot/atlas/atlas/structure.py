"""Validate the JSON Schema subset used by the project's own schemas.

This is deliberately not advertised as a general JSON Schema implementation.
Consumers may use a draft-2020-12 validator instead; semantic model validation
is always additionally required.
"""
import math
import re
from .dsl import ValidationError


def check(value,schema,path='$'):
    if 'oneOf' in schema:
        matched=0
        for option in schema['oneOf']:
            try:check(value,option,path);matched+=1
            except ValidationError:pass
        if matched!=1:raise ValidationError(f'{path}: expected exactly one allowed representation')
    if 'const' in schema and value!=schema['const']:raise ValidationError(f'{path}: invalid constant')
    if 'enum' in schema and value not in schema['enum']:raise ValidationError(f'{path}: value not in enum')
    typ=schema.get('type')
    good={'object':isinstance(value,dict),'array':isinstance(value,list),'string':isinstance(value,str),
          'number':type(value) in (int,float) and math.isfinite(value) if type(value) in (int,float) else False,
          'integer':type(value) is int,'boolean':type(value) is bool}
    if typ and not good.get(typ,False):raise ValidationError(f'{path}: expected {typ}')
    if isinstance(value,dict):
        if not set(schema.get('required',[]))<=value.keys():raise ValidationError(f'{path}: required fields missing')
        if len(value)<schema.get('minProperties',0):raise ValidationError(f'{path}: too few fields')
        properties=schema.get('properties',{});additional=schema.get('additionalProperties',True)
        for key,item in value.items():
            if 'propertyNames' in schema:check(key,schema['propertyNames'],path+'.<key>')
            if key in properties:check(item,properties[key],path+'.'+key)
            elif additional is False:raise ValidationError(f'{path}: unknown field {key}')
            elif isinstance(additional,dict):check(item,additional,path+'.'+key)
    if isinstance(value,list):
        if len(value)<schema.get('minItems',0) or len(value)>schema.get('maxItems',math.inf):raise ValidationError(f'{path}: invalid array length')
        for i,item in enumerate(value):check(item,schema.get('items',{}),f'{path}[{i}]')
    if isinstance(value,str):
        if len(value)<schema.get('minLength',0) or len(value)>schema.get('maxLength',math.inf):raise ValidationError(f'{path}: invalid string length')
        if 'pattern' in schema and not re.search(schema['pattern'],value):raise ValidationError(f'{path}: invalid string pattern')
    if type(value) in (float,int):
        if 'exclusiveMinimum' in schema and value<=schema['exclusiveMinimum']:raise ValidationError(f'{path}: value must exceed minimum')
        if value<schema.get('minimum',-math.inf) or value>schema.get('maximum',math.inf):raise ValidationError(f'{path}: number outside bounds')
