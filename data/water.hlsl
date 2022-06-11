// ---------
// CONSTANTS
// ---------

cbuffer CONSTANT_BUFFER : register(b0)
{
	float4x4 view_and_projection;
	float water_level;
	float time;
};

float2 random2(float2 p)
{
	return frac(sin(float2(dot(p,float2(127.1,311.7)),dot(p,float2(269.5,183.3))))*43758.5453);
}

// -------------
// VERTEX SHADER
// -------------

struct VS_INPUT
{
	float3 position : POSITION;
	float3 normal : NORMAL;
	float2 texcoord: TEXCOORD0;
};

struct VS_OUTPUT
{
	float3 vertex_position: POSITION;
	float4 position: SV_POSITION;
	float3 normal : NORMAL;
	float2 texcoord: TEXCOORD0;
};

VS_OUTPUT VSMain(VS_INPUT input)
{
	VS_OUTPUT output;

	float4 v = float4(input.position + float3(0,water_level,0), 1);
	v.y -= sin(time);

	output.position = mul(view_and_projection, v);
	output.vertex_position = input.position;
	output.normal = input.normal;
	output.texcoord = input.texcoord;
	
	return output;
}

// ------------
// PIXEL SHADER
// ------------

Texture2D<uint2> depthbuffer : register(t0);

struct PS_INPUT
{
	float3 vertex_position: POSITION;
	float4 position: SV_POSITION;
	float3 normal : NORMAL;
	float2 texcoord: TEXCOORD0;
};

struct PS_OUTPUT
{
	float4 color;
};

PS_OUTPUT PSMain(PS_INPUT input) : SV_TARGET
{
	PS_OUTPUT output;

	float3 color = 0;

#if 1
	float2 tc = input.texcoord * 200;
	float2 i_st = floor(tc);
	float2 f_st = frac(tc);

	float m_dist = 1;

	for(int y = -1; y <= 1; y++)
	{
		for(int x = -1; x <= 1; x++)
		{
			float2 neighbor = float2(float(x),float(y));

			float2 p = random2(neighbor + i_st);
			p = 0.5 + 0.5 * sin(time + 6.2831*p);

			float2 diff = neighbor + p - f_st;

			float dist = length(diff);
			m_dist = min(m_dist, dist);
		}
	}

	color += m_dist;
#endif

	color += float3(0.0, .2, .5);
	output.color = float4(color,0.5);

	return output;
}
