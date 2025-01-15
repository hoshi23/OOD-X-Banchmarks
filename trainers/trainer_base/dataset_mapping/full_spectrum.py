FULL_SPECTRUM_MAPPING: dict[str, dict] = {
    "ImageNetHard": {
        "id": {
            "s": ["ImageNetHard"],
            "fs": [
                "ImageNetHard",
                "ImageNetV2Hard",
                "ImageNetRHard",
                "ImageNetCHard",
            ],
        },
        "ood": {
            "s": ["ImageNetHard"],
            "fs": [
                "ImageNetHard",
                "ImageNetV2Hard",
                "ImageNetRHard",
                "ImageNetCHard",
            ],
        },
    },
    "IWildCamHard": {
        "id": {
            "s": ["IWildCamIdTestHard"],
            "fs": [
                "IWildCamIdTestHard",
                "IWildCamHard",
            ],
        },
        "ood": {
            "s": ["IWildCamIdTestHard"],
            "fs": [
                "IWildCamIdTestHard",
                "IWildCamHard",
            ],
        },
    },
    "FMoWHard": {
        "id": {
            "s": ["FMoWIdTestHard"],
            "fs": [
                "FMoWIdTestHard",
                "FMoWHard",
            ],
        },
        "ood": {
            "s": ["FMoWIdTestHard"],
            "fs": [
                "FMoWIdTestHard",
                "FMoWHard",
            ],
        },
    },
}
