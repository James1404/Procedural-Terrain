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

cbuffer VS_CONSTANT_BUFFER : register(b0)
{
	float4x4 view_and_projection;
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
TextureCube cubemap_texture;
SamplerState sampler_type;

struct PS_INPUT
{
	float4 position: SV_POSITION;
	float3 texcoord: TEXCOORD0;
};

struct PS_OUTPUT
{
	float4 color;
};

PS_OUTPUT PSMain(PS_INPUT input) : SV_TARGET
{
	PS_OUTPUT output;

	float4 color = cubemap_texture.Sample(sampler_type, input.texcoord);

	//color = float4(1.0, 0.4, 0.1, 0.0);

	output.color = color;

	return output;
}
