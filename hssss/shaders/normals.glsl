#version 330 core

in vec3 vNormal;
in vec3 vLight;

uniform vec3 uColor;     // base color

out vec4 fragColor;

void main()
{
    vec3 N = normalize(vNormal);

    vec3 display = 0.5*N + vec3(0.5, 0.5, 0.5);

    fragColor = vec4(display, 1.0);
}