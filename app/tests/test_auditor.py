from app.src.auditor.core import Auditor
from app.src.auditor.models import ActivityModel
from app.src.injector.core import Injector
from app.src.injector.enums import GlobalTypes
from app.src.config.core import Config
from app.src.network.core import Network
import pytest

@pytest.fixture
def auditor() -> Auditor:
    Injector.register(name=GlobalTypes.CONFIG, instance=Config())
    return Auditor()

@pytest.fixture(autouse=True)
def reset_injector():
    Injector._instances.clear()
    yield
    Injector._instances.clear()

def test_auditor_doesComputeNeuronGrowth(auditor: Auditor):
    auditor = Auditor()
    auditor.activity = ActivityModel(
            features=20,
            new_sensors=3,
            total_sensors=20,
            new_motors=20,
            total_motors=3,
            nexus_size=100,
            confidence=[0.9]
        )
    
    result = auditor.determine_growth(
        reward=0.1,
    )
    assert result.nexus_growth > 0
    assert result.aperture_growth == 0
    assert result.terminus_growth > 0

def test_auditor_doesComputeNeuronGrowthForSmallNetwork(auditor: Auditor):
    auditor.activity = ActivityModel(
        features=20,
        new_sensors=9,
        total_sensors=9,
        new_motors=2,
        total_motors=2,
        nexus_size=3,
        confidence=[0.9]
    )
    result = auditor.determine_growth(
        reward=0.1
    )
    assert result.nexus_growth > 0
    assert result.aperture_growth == 0
    assert result.terminus_growth == 0