#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_FILE="${PROJECT_ROOT}/bench_results.zip"
TARGET_DIRS=("nsys-reps" "ncu-reps" "logs")

if ! command -v zip >/dev/null 2>&1; then
	echo "ERROR: zip command not found. Please install zip first." >&2
	exit 1
fi

echo "Preparing target directories..."
for dir_name in "${TARGET_DIRS[@]}"; do
	mkdir -p "${PROJECT_ROOT}/${dir_name}"
done

if [[ -f "${ARCHIVE_FILE}" ]]; then
	timestamp="$(date +"%Y%m%d_%H%M%S")"
	backup_file="${PROJECT_ROOT}/bench_results_${timestamp}.zip"
	suffix=1

	while [[ -e "${backup_file}" ]]; do
		backup_file="${PROJECT_ROOT}/bench_results_${timestamp}_${suffix}.zip"
		suffix=$((suffix + 1))
	done

	mv "${ARCHIVE_FILE}" "${backup_file}"
	echo "Backed up existing archive to: ${backup_file}"
fi

echo "Creating archive: ${ARCHIVE_FILE}"
(
	cd "${PROJECT_ROOT}"
	zip -rq "${ARCHIVE_FILE}" "${TARGET_DIRS[@]}"
)

if [[ ! -f "${ARCHIVE_FILE}" ]]; then
	echo "ERROR: Failed to create archive: ${ARCHIVE_FILE}" >&2
	exit 1
fi

archive_size_human="$(du -h "${ARCHIVE_FILE}" | awk '{print $1}')"
archive_size_bytes="$(wc -c < "${ARCHIVE_FILE}")"

echo "Archive created successfully: ${ARCHIVE_FILE}"
echo "Archive size: ${archive_size_human} (${archive_size_bytes} bytes)"
echo "Included directories: ${TARGET_DIRS[*]}"

exit 0
