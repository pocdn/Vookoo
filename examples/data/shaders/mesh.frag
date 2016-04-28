#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 normal_in;
layout (location = 1) in vec3 eyeDirection_in;
layout (location = 2) in vec3 lightDirection_in;

layout (location = 0) out vec4 fragColor_out;

void main() 
{
  /*vec3 Eye = normalize(-inEyePos);
  vec3 Reflected = normalize(reflect(-inLightVec, inNormal)); 
 
  vec4 IAmbient = vec4(0.2, 0.2, 0.2, 1.0);
  vec4 IDiffuse = vec4(0.5, 0.5, 0.5, 0.5) * max(dot(inNormal, inLightVec), 0.0);
  float specular = 0.25;
  vec4 ISpecular = vec4(0.5, 0.5, 0.5, 1.0) * pow(max(dot(Reflected, Eye), 0.0), 0.8) * specular; 
 
  outFragColor = vec4((IAmbient + IDiffuse) * vec4(inColor, 1.0) + ISpecular);
  */
  fragColor_out = vec4(normal_in, 1.0);
}
