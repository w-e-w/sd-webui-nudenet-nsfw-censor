import modules.script_callbacks as script_callbacks
import scripts.nudenet_nsfw_censor_scripts.settings
from scripts.nudenet_nsfw_censor_scripts.processing_script import ScriptNudenetCensor
from scripts.nudenet_nsfw_censor_scripts.post_processing_script import ScriptPostprocessingNudenetCensor
import scripts.nudenet_nsfw_censor_scripts.api as api
script_callbacks.on_app_started(api.nudenet_censor_api)
