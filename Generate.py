import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

try:
    # Load Stable Diffusion model with optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16  # Using FP16 to reduce VRAM usage
    ).to("cuda")

    # Apply memory-saving techniques
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    # Define the prompt for image generation
    prompt = "A man riding a Kawasaki bike on a race track"
    negative_prompt = "blurry, low-resolution, unrealistic, distorted"

    # Clear GPU memory before execution
    torch.cuda.empty_cache()

    # Generate an image using optimized settings for 4GB GPUs
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,  # Balanced for speed and quality
        guidance_scale=7.5,
        height=512, width=512  # Optimized for low VRAM
    ).images[0]

    # Save the generated image
    image.save("optimized_image.png")
    print("✅ Image saved successfully!")

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axis for better viewing
    plt.show()

except Exception as e:
    print(f"❌ Error occurred: {str(e)}")
    print("Try reinstalling dependencies: pip install --upgrade torch torchvision transformers diffusers")
    print("Also, ensure there are no conflicting local files named 'torch.py' or 'diffusers.py'.")