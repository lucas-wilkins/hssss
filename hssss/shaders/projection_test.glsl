#version 330 core

in vec3 vNormal;
in vec3 vWorldPos;

uniform mat4 uProjectorMVP;
uniform mat4 uProjectorModel;

uniform vec3 uColor;

out vec4 fragColor;

void main()
{

    /* Shows x and y of the projector as red and green */

    vec4 lightSpacePos = uProjectorMVP * vec4(vWorldPos, 1.0);
    vec2 uv = lightSpacePos.xy / lightSpacePos.w;

    uv = uv * 0.5 + 0.5;
    uv = mod(uv, 1.0);

    fragColor = vec4(uv, 0, 0);
}