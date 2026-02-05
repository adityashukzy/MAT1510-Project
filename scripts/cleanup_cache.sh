#!/bin/bash
# Helper script to manage disk space on the cluster

echo "============================================"
echo "Disk Usage Summary"
echo "============================================"

echo ""
echo "Cache sizes:"
du -sh ~/.cache/* 2>/dev/null | sort -h

echo ""
echo "Total cache:"
du -sh ~/.cache 2>/dev/null

echo ""
echo "HuggingFace models:"
ls -lh ~/.cache/huggingface/hub/ 2>/dev/null | grep ^d | awk '{print $9, $5}'

echo ""
echo "============================================"
echo "Cleanup Options"
echo "============================================"
echo ""
echo "To clear UV cache (safe, will rebuild on next job):"
echo "  rm -rf ~/.cache/uv"
echo ""
echo "To clear specific HuggingFace model (e.g., old models):"
echo "  rm -rf ~/.cache/huggingface/hub/models--<model-name>"
echo ""
echo "To clear ALL HuggingFace cache (WARNING: will re-download models):"
echo "  rm -rf ~/.cache/huggingface"
echo ""
echo "To clear pip cache if it exists:"
echo "  rm -rf ~/.cache/pip"
echo ""
