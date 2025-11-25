import json
import sys
from typing import List

from .models import AuditResult


class NDJSONOutputter:

    def output_results(self, results: List[AuditResult]) -> None:
        # output results as NDJSON to stdout
        for result in results:
            self.output_single_result(result)

    def output_single_result(self, result: AuditResult) -> None:
        data = {
            "name": result.name,
            "category": result.category,
            "net_score": result.net_score,
            "net_score_latency": result.net_score_latency,
            "ramp_up_time": result.ramp_up_time,
            "ramp_up_time_latency": result.ramp_up_time_latency,
            "bus_factor": result.bus_factor,
            "bus_factor_latency": result.bus_factor_latency,
            "performance_claims": result.performance_claims,
            "performance_claims_latency": result.performance_claims_latency,
            "license": result.license,
            "license_latency": result.license_latency,
            "size_score": {
                "raspberry_pi": result.size_score.raspberry_pi,
                "jetson_nano": result.size_score.jetson_nano,
                "desktop_pc": result.size_score.desktop_pc,
                "aws_server": result.size_score.aws_server,
            },
            "size_score_latency": result.size_score_latency,
            "dataset_and_code_score": result.dataset_and_code_score,
            "dataset_and_code_score_latency": result.dataset_and_code_score_latency,
            "dataset_quality": result.dataset_quality,
            "dataset_quality_latency": result.dataset_quality_latency,
            "code_quality": result.code_quality,
            "code_quality_latency": result.code_quality_latency,
        }
        json_line = json.dumps(data, separators=(",", ":"))
        print(json_line, file=sys.stdout)
        sys.stdout.flush()
