#version 330 core

in vec3 vNormal;
in vec3 vLight;

uniform vec3 uColor;     // base color

out vec4 fragColor;


void main()
{
    vec3 L = normalize(vLight);

    vec3 display = 0.5*L + vec3(0.5, 0.5, 0.5);
//    vec3 display = vec3(0.5, 0.5, 0.5);

    fragColor = vec4(display, 1.0);
}