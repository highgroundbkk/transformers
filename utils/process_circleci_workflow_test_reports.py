# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import re
from collections import Counter

import requests


def _extract_model_name(test_node_id):
    """Extract model name from test path like tests/models/bart/..."""
    file_path = test_node_id.split("::", 1)[0]
    if file_path.startswith("tests/models/"):
        parts = file_path.split("/")
        if len(parts) >= 3:
            return parts[2]
    return None


def _normalize_test_name(test_name):
    """Normalize test name by removing parametrization and numeric suffixes."""
    base_name = test_name.split("[", 1)[0]
    parts = base_name.split("::")
    if parts:
        parts[-1] = re.sub(r"_\d{2,}.*$", "", parts[-1])
    return "::".join(parts)


def _aggregate_failures(failure_entries):
    """Aggregate failures by test and by model."""
    by_test = {}
    by_model = {}

    for entry in failure_entries:
        test_name = entry["test_name"]
        model_name = entry["model_name"]
        error = entry["error"]
        normalized = _normalize_test_name(test_name)

        # Aggregate by test
        if normalized not in by_test:
            by_test[normalized] = {"count": 0, "errors": Counter()}
        by_test[normalized]["count"] += 1
        by_test[normalized]["errors"][error] += 1

        # Aggregate by model
        if model_name:
            if model_name not in by_model:
                by_model[model_name] = {"count": 0, "errors": Counter()}
            by_model[model_name]["count"] += 1
            by_model[model_name]["errors"][error] += 1

    # Convert Counter objects to dicts for JSON serialization
    for info in by_test.values():
        info["errors"] = dict(info["errors"].most_common())
    for info in by_model.values():
        info["errors"] = dict(info["errors"].most_common())

    return by_test, by_model


def _create_markdown_report(failures_by_test, failures_by_model):
    """Create markdown summary of failures."""
    lines = ["# Failure summary\n"]

    if failures_by_test:
        lines.append("## By test\n")
        lines.append("| Test | Failures | Error(s) |")
        lines.append("| --- | --- | --- |")
        for test, info in sorted(failures_by_test.items(), key=lambda x: x[1]["count"], reverse=True):
            errors = "; ".join(f"{count}× {msg}" for msg, count in info["errors"].items())
            lines.append(f"| {test} | {info['count']} | {errors} |")
        lines.append("")

        lines.append("## By model\n")
        lines.append("| Model | Failures | Error(s) |")
        lines.append("| --- | --- | --- |")
        for model, info in sorted(failures_by_model.items(), key=lambda x: x[1]["count"], reverse=True):
            errors = "; ".join(f"{count}× {msg}" for msg, count in info["errors"].items())
            lines.append(f"| {model} | {info['count']} | {errors} |")
    else:
        lines.append("No failures were reported.\n")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_id", type=str, required=True)
    args = parser.parse_args()
    workflow_id = args.workflow_id

    r = requests.get(
        f"https://circleci.com/api/v2/workflow/{workflow_id}/job",
        headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")},
    )
    jobs = r.json()["items"]

    os.makedirs("outputs", exist_ok=True)

    workflow_summary = {}
    failure_entries = []

    # for each job, download artifacts
    for job in jobs:
        project_slug = job["project_slug"]
        if job["name"].startswith(("tests_", "examples_", "pipelines_")):
            url = f"https://circleci.com/api/v2/project/{project_slug}/{job['job_number']}/artifacts"
            r = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
            job_artifacts = r.json()["items"]

            os.makedirs(f"outputs/{job['name']}", exist_ok=True)

            job_test_summaries = {}
            job_failure_lines = {}

            for artifact in job_artifacts:
                node_index = artifact["node_index"]
                url = artifact["url"]

                if artifact["path"].startswith("reports/") and artifact["path"].endswith("/summary_short.txt"):
                    r = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    job_test_summaries[node_index] = r.text
                elif artifact["path"].startswith("reports/") and artifact["path"].endswith("/failures_line.txt"):
                    r = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    job_failure_lines[node_index] = r.text

            summary = {}
            for node_index, node_test_summary in job_test_summaries.items():
                for line in node_test_summary.splitlines():
                    if line.startswith("PASSED "):
                        test = line[len("PASSED "):]
                        summary[test] = "passed"
                    elif line.startswith("FAILED "):
                        test = line[len("FAILED "):].split()[0]
                        summary[test] = "failed"

            # failed before passed
            summary = dict(sorted(summary.items(), key=lambda x: (x[1], x[0])))
            workflow_summary[job["name"]] = summary

            # collected version
            with open(f"outputs/{job['name']}/test_summary.json", "w") as fp:
                json.dump(summary, fp, indent=4)

            # Collect failure details
            for node_index, summary_text in job_test_summaries.items():
                failure_lines_list = []
                if node_index in job_failure_lines:
                    for line in job_failure_lines[node_index].splitlines():
                        line = line.strip()
                        if line and not line.startswith(("=", "_")) and ": " in line:
                            if not line.lower().startswith("short test summary"):
                                failure_lines_list.append(line)

                failure_idx = 0
                for line in summary_text.splitlines():
                    if line.startswith("FAILED ") and " - Failed: (subprocess)" not in line:
                        failure_line = line[len("FAILED "):].strip()
                        test_name, _, short_error = failure_line.partition(" - ")
                        full_error = failure_lines_list[failure_idx] if failure_idx < len(
                            failure_lines_list) else short_error
                        failure_entries.append({
                            "job_name": job["name"],
                            "test_name": test_name.strip(),
                            "error": full_error,
                            "model_name": _extract_model_name(test_name.strip()),
                        })
                        failure_idx += 1

    new_workflow_summary = {}
    for job_name, job_summary in workflow_summary.items():
        for test, status in job_summary.items():
            if test not in new_workflow_summary:
                new_workflow_summary[test] = {}
            new_workflow_summary[test][job_name] = status

    for test, result in new_workflow_summary.items():
        new_workflow_summary[test] = dict(sorted(result.items()))
    new_workflow_summary = dict(sorted(new_workflow_summary.items()))

    with open("outputs/test_summary.json", "w") as fp:
        json.dump(new_workflow_summary, fp, indent=4)

    # Create failure summary
    failures_by_test, failures_by_model = _aggregate_failures(failure_entries)

    failure_summary = {
        "failures": failure_entries,
        "by_test": failures_by_test,
        "by_model": failures_by_model,
    }

    with open("outputs/failure_summary.json", "w") as fp:
        json.dump(failure_summary, fp, indent=4)

    markdown_report = _create_markdown_report(failures_by_test, failures_by_model)
    with open("outputs/failure_summary.md", "w") as f:
        f.write(markdown_report)