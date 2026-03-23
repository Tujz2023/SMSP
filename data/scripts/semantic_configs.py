configs = [
    {
        "name": "chinese_architecture",
        "prompt": "A highly detailed traditional Chinese pavilion and pagoda architecture, with elegant wooden beams and curved roofs, ornate carvings and stone pathways, set beside a calm lake. In the background is a bright blue sky with fluffy white clouds, soft sunlight casting shadows. On the water are small wooden boats with people rowing peacefully. Cinematic lighting, ultra-realistic, high resolution.",
        "guidance_scale": 9,
        "num_inference_steps": 50,
        "control_guidance_end": 0.75,
        "scheduler": "EulerAncestralDiscreteScheduler",
        "enhance": 0.7,
        "invert": False
    },
    {
        "name": "cyberpunk_city",
        "prompt": "A sprawling futuristic cyberpunk city at night, rainy street with wet asphalt reflecting neon signs. Neon lights in cyan, magenta, purple, deep reflections and glows. Cinematic moody lighting, high dynamic contrast, photorealistic, ultra-detailed environment — a vibrant, chaotic scene. Intricate details in architecture and street elements, bustling urban atmosphere, high-resolution.",
        "guidance_scale": 9,
        "num_inference_steps": 50,
        "control_guidance_end": 0.8,
        "scheduler": "EulerAncestralDiscreteScheduler",
        "enhance": 0.45,
        "invert": True
    },
    {
        "name": "winter_valley",
        "prompt": "A vast winter mountain valley covered in fresh snow, snow-covered pine trees, frosty ice crystals on branches, an almost-frozen lake partially reflecting snowy peaks, distant snow-covered mountains under pale winter sunlight. Ultra-detailed, photorealistic landscape, high resolution, crisp snow and ice textures, realistic lighting and shading, cinematic winter serenity.",
        "guidance_scale": 9,
        "num_inference_steps": 50,
        "control_guidance_end": 0.7,
        "scheduler": "EulerAncestralDiscreteScheduler",
        "enhance": 0.8,
        "invert": False
    }
]