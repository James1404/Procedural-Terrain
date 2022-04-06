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
	float4 position: SV_POSITION;
	float3 normal : NORMAL;
	float2 texcoord: TEXCOORD0;
};

cbuffer VS_CONSTANT_BUFFER : register(b0)
{
	float4x4 view_and_projection;
	float4x4 transform;
};

VS_OUTPUT VSMain(VS_INPUT input)
{
	VS_OUTPUT output;

	//output.position = mul(float4(input.position, 1.0), transform);
	//output.position = mul(output.position, view_and_projection);

	output.position = mul(view_and_projection, mul(transform, float4(input.position, 1.0)));
	output.normal = input.normal;
	output.texcoord = input.texcoord;
	
	return output;
}

// ------------
// PIXEL SHADER
// ------------
Texture2D shader_texture;
SamplerState sampler_type;

struct PS_INPUT
{
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

	output.color = shader_texture.Sample(sampler_type, input.texcoord);

	return output;
}
