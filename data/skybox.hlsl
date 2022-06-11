// ---------
// CONSTANTS
// ---------

cbuffer CONSTANT_BUFFER : register(b0)
{
	float4x4 view_and_projection;
	float time;
};

// -------------
// VERTEX SHADER
// -------------

struct VS_INPUT
{
	float3 position : POSITION;
	float2 texcoord: TEXCOORD0;
};

struct VS_OUTPUT
{
	float4 position: SV_POSITION;
	float3 texcoord: TEXCOORD0;
};

VS_OUTPUT VSMain(VS_INPUT input)
{
	VS_OUTPUT output;

	float4 pos = mul(view_and_projection, float4(input.position, 1.0));
	output.position = pos.xyww;
	output.texcoord = input.position;
	
	return output;
}

// ------------
// PIXEL SHADER
// ------------

struct PS_INPUT
{
	float4 position: SV_POSITION;
	float2 texcoord: TEXCOORD0;
};

struct PS_OUTPUT
{
	float4 color;
};

float random (float2 st)
{
    return frac(sin(dot(st.xy,float2(12.9898,78.233)))*43758.5453123);
}

float noise (float2 st)
{
    float2 i = floor(st);
    float2 f = frac(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + float2(1.0, 0.0));
    float c = random(i + float2(0.0, 1.0));
    float d = random(i + float2(1.0, 1.0));

    float2 u = f * f * (3.0 - 2.0 * f);

    return lerp(a, b, u.x) +
               (c - a)* u.y * (1.0 - u.x) +
               (d - b) * u.x * u.y;
}

#define CLOUD_SCALE 10

#define OCTAVES 6
float fbm (float2 st) {
    float value = 0.048;
    float amplitude = .8;

    for (int i = 0; i < OCTAVES; i++)
	{
        value += amplitude * noise(st);
        st *= 2.;
        amplitude *= .5;
    }
    return value;
}

#define SKY_TOP_COLOUR 	  float4(.106, .106, .431, 1)
#define SKY_BOTTOM_COLOUR float4(.275, .431, .639, 1)

#define CLOUD_COLOR       float4(.9,.1,.2,1)

PS_OUTPUT PSMain(PS_INPUT input) : SV_TARGET
{
	PS_OUTPUT output;

	float4 color = lerp(SKY_BOTTOM_COLOUR,SKY_TOP_COLOUR,input.texcoord.y);


	float cloud_blend = 1.5;
	float cloud_level = 0.6;
	float blend_value = smoothstep(cloud_level-cloud_blend,
								   cloud_level+cloud_blend,
								   input.texcoord.y);

	float v = fbm(input.texcoord*CLOUD_SCALE);
	float4 clouds = v;
	clouds *= CLOUD_COLOR;
	clouds.a = 1;


	color = lerp(color, clouds, blend_value);

	output.color = color;

	return output;
}
