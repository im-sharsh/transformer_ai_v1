"""
Manifest: the audit trail of every automated decision the pipeline made.
This is what turns "automated" into "automated and trustworthy."
"""
import json
import datetime


class Manifest:
    def __init__(self, source_name: str):
        self.data = {
            "source_name": source_name,
            "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "schema_resolution": {},
            "cleaning": {},
            "shaping": {},
            "warnings": [],
        }

    def set_schema_resolution(self, mapping_report, quarantined_columns):
        self.data["schema_resolution"] = {
            "mapping": mapping_report,
            "quarantined_columns": quarantined_columns,
        }

    def set_cleaning(self, stats: dict):
        self.data["cleaning"] = stats

    def set_shaping(self, task_type: str, stats: dict):
        self.data["shaping"][task_type] = stats

    def warn(self, message: str):
        self.data["warnings"].append(message)

    def to_dict(self):
        return self.data

    def to_json(self, indent=2):
        return json.dumps(self.data, indent=indent, default=str)
