#version 330 core

in vec3 vNormal;
in vec3 vWorldPos;

uniform mat4 uProjectorMVP;
uniform mat4 uProjectorModelView;

uniform sampler2D uProjectionTexture;

uniform vec3 uColor;     // base color

out vec4 fragColor;

void main()
{

    vec4 lightSpacePos = uProjectorMVP * vec4(vWorldPos, 1.0);
    vec2 uv = lightSpacePos.xy / lightSpacePos.w;

    uv = uv * 0.5 + 0.5;

    float scale = texture(uProjectionTexture, uv).r;

    fragColor = vec4(scale * uColor, 1.0);
}