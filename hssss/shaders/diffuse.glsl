#version 330 core

in vec3 vNormal;

uniform vec3 uColor;

out vec4 fragColor;

void main()
{

    vec3 N = normalize(vNormal);
    vec3 L = normalize(vLight);

    float diff = max(dot(N, L), 0.0);

    fragColor = vec4(uColor * diff, 1.0);
}