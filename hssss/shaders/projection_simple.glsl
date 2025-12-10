#version 330 core

in vec3 vNormal;
in vec3 vWorldPos;

uniform vec3 uColor;

uniform mat4 uProjectorMVP;
uniform vec3 uProjectorPos;

uniform sampler2D uProjectionTexture;

out vec4 fragColor;


void main()
{

    vec4 lightSpacePos = uProjectorMVP * vec4(vWorldPos, 1.0);

    vec3 lightSpacePosModel = uProjectorPos - vWorldPos;

    vec2 uv = lightSpacePos.xy / lightSpacePos.w;

    uv = uv * 0.5 + 0.5;

    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightSpacePosModel);

    float diff = max(dot(N, L), 0.0);

    float scale = texture(uProjectionTexture, uv).r;

    fragColor = vec4(diff * scale * uColor, 1.0);

}