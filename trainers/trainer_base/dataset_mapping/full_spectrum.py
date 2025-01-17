FULL_SPECTRUM_MAPPING: dict[str, dict] = {
    "ImageNetX": {
        "id": {
            "s": ["ImageNetX"],
            "fs": [
                "ImageNetX",
                "ImageNetV2X",
                "ImageNetRX",
                "ImageNetCX",
            ],
        },
        "ood": {
            "s": ["ImageNetX"],
            "fs": [
                "ImageNetX",
                "ImageNetV2X",
                "ImageNetRX",
                "ImageNetCX",
            ],
        },
    },
    "IWildCamX": {
        "id": {
            "s": ["IWildCamIdTestX"],
            "fs": [
                "IWildCamIdTestX",
                "IWildCamX",
            ],
        },
        "ood": {
            "s": ["IWildCamIdTestX"],
            "fs": [
                "IWildCamIdTestX",
                "IWildCamX",
            ],
        },
    },
    "FMoWX": {
        "id": {
            "s": ["FMoWIdTestX"],
            "fs": [
                "FMoWIdTestX",
                "FMoWX",
            ],
        },
        "ood": {
            "s": ["FMoWIdTestX"],
            "fs": [
                "FMoWIdTestX",
                "FMoWX",
            ],
        },
    },
}
