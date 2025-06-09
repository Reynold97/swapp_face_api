from typing import List, Optional, Union, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, conlist, conint, confloat, HttpUrl


class SwapMode(str, Enum):
    """Enum for swap modes."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    SORTED = "sorted"
    SIMILARITY = "similarity"


class Direction(str, Enum):
    """Enum for sorting direction in sorted mode."""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"


class ProcessingOptions(BaseModel):
    """Common processing options for face swapping."""
    mode: SwapMode = Field(
        default=SwapMode.ONE_TO_ONE,
        description="Face swap mode"
    )
    direction: Direction = Field(
        default=Direction.LEFT_TO_RIGHT,
        description="Direction for sorted mode (only used in sorted mode)"
    )
    use_codeformer: bool = Field(
        default=False,
        description="Whether to use CodeFormer for enhancement"
    )
    codeformer_fidelity: float = Field(
        default=0.5,
        description="Balance between quality and fidelity for CodeFormer",
        ge=0.0,
        le=1.0
    )
    background_enhance: bool = Field(
        default=True,
        description="Whether to enhance the background (CodeFormer only)"
    )
    face_upsample: bool = Field(
        default=True,
        description="Whether to upsample the faces (CodeFormer only)"
    )
    upscale: int = Field(
        default=2,
        description="The upscale factor for enhancement",
        ge=1,
        le=4
    )
    face_refinement_steps: int = Field(
        default=1,
        description="Number of repeated face swaps to refine the result (1-5)",
        ge=1,
        le=5
    )

    @field_validator('upscale')
    @classmethod  # Keep the classmethod decorator
    def validate_upscale(cls, v):
        """Validate upscale factor is within reasonable limits."""
        if v > 4:
            return 4  # Cap at 4 to prevent excessive memory usage
        return v

    @field_validator('codeformer_fidelity')
    @classmethod
    def validate_fidelity(cls, v):
        """Validate codeformer_fidelity is between 0 and 1."""
        return max(0.0, min(1.0, v))  # Clamp between 0 and 1
    
    # New validator for face_refinement_steps (optional since we have ge/le constraints)
    @field_validator('face_refinement_steps')
    @classmethod
    def validate_swap_iterations(cls, v):
        """Validate swap_iterations is within reasonable limits."""
        return max(1, min(5, v))  # Clamp between 1 and 5


class SwapUrlRequest(BaseModel):
    """Request schema for the swap_url endpoint."""
    model_filenames: List[str] = Field(
        ..., 
        description="List of filenames for target images in the GCP bucket",
        min_items=1
    )
    face_filename: str = Field(
        ..., 
        description="Filename for the source face image in the GCP bucket"
    )
    options: ProcessingOptions = Field(
        default_factory=ProcessingOptions,
        description="Processing options for face swapping"
    )


class SwapImgRequest(BaseModel):
    """Request schema for the swap_img endpoint (just the options part, files will be handled separately)."""
    options: ProcessingOptions = Field(
        default_factory=ProcessingOptions,
        description="Processing options for face swapping"
    )


class FailedModel(BaseModel):
    """Schema for failed model information."""
    filename: str = Field(..., description="Filename of the failed model")
    error: str = Field(..., description="Error message")


class SuccessResponse(BaseModel):
    """Schema for successful response."""
    urls: List[str] = Field(..., description="List of URLs for processed images")


class PartialSuccessResponse(SuccessResponse):
    """Schema for partially successful response."""
    failed: List[FailedModel] = Field(..., description="List of failed models")


class ErrorResponse(BaseModel):
    """Schema for error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[FailedModel]] = Field(None, description="Detailed error information")