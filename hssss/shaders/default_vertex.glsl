#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;      // needed for transforming normals
uniform vec3 uLightDir;

out vec3 vNormal;
out vec3 vLight;
out vec3 vWorldPos;

void main()
{

    gl_Position = uMVP * vec4(aPos, 1.0);

    // Transform normal to world space
    vNormal = mat3(uModel) * aNormal;

    // Pass light direction
    vLight = uLightDir;

    // Pass world position
    vWorldPos = aPos;
}