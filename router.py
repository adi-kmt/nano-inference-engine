from fastapi import APIRouter, Query, Depends, HTTPException, BackgroundTasks
from dependency_container import get_engine
from dtos import CompletionRequest
from engine.engine import Engine
import asyncio

router = APIRouter()

# Track loading status
model_loading_status = {"loading": False, "loaded": False, "error": None}


@router.post("/load_model")
async def load_model(
        model_name: str = Query(..., description="Name of the model to load"),
        background_tasks: BackgroundTasks = None,
        engine: Engine = Depends(get_engine)
):
    global model_loading_status

    if model_loading_status["loading"]:
        raise HTTPException(400, "Model is already loading")

    if engine.model is not None:
        raise HTTPException(400, "Model already loaded. Shutdown first.")

    # Reset status
    model_loading_status = {"loading": True, "loaded": False, "error": None}

    def load_model_sync():
        try:
            engine.load_model(model_name)
            model_loading_status.update({"loading": False, "loaded": True})
            print(f"Model {model_name} loaded successfully in background")
        except Exception as e:
            model_loading_status.update({"loading": False, "error": str(e)})
            print(f"Failed to load model in background: {e}")

    # Run in thread pool to avoid blocking
    await asyncio.get_event_loop().run_in_executor(None, load_model_sync)

    return {
        "message": f"Model {model_name} started loading in background",
        "status": "loading"
    }


@router.get("/load_status")
async def get_load_status():
    return model_loading_status


@router.post("/unload_model")
async def unload_model(engine: Engine = Depends(get_engine)):
    global model_loading_status

    if model_loading_status["loading"]:
        raise HTTPException(400, "Model is currently loading, cannot unload")

    engine.model = None
    engine.tokenizer = None
    model_loading_status = {"loading": False, "loaded": False, "error": None}

    return {"message": "Model unloaded successfully"}


@router.post("/v1/completions")
async def generate(
        request: CompletionRequest,
        engine: Engine = Depends(get_engine)
):
    if not model_loading_status["loaded"]:
        if model_loading_status["loading"]:
            raise HTTPException(400, "Model is still loading, please wait")
        else:
            raise HTTPException(400, "No model loaded. Please load a model first.")

    try:
        output = engine.generate(request.prompt)
        return {"generated_text": output}
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")