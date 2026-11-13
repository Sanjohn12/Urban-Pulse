"""
Automated Screenshot Capture for Research Figures 13 to 19
Captures high-resolution publication screenshots of the running Streamlit app:
  [x] Figure 13 – Main dashboard and study area map
  [x] Figure 14 – Interactive grid selection and information panel
  [x] Figure 15 – Urban activity prediction and 24-hour pulse
  [x] Figure 16 – SHAP explanation and AI-generated response
  [x] Figure 17 – Theory-guided interpretation tab (Mode 4 TSE-UI)
  [x] Figure 18 – What-if scenario analysis
  [x] Figure 19 – Multi-agent workflow or execution log

Does NOT modify any existing application code.
"""

import sys
import time
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

from project_paths import PROJECT_ROOT, FIGURES_DIR
WORKSPACE_DIR = PROJECT_ROOT
OUTPUT_DIR = FIGURES_DIR
OUTPUT_DIR.mkdir(exist_ok=True)
from project_paths import PYTHON_EXECUTABLE
PYTHON_EXE = PYTHON_EXECUTABLE

print("=" * 80)
print("AUTOMATED SCREENSHOT CAPTURE (Figures 13 - 19)")
print("Output Directory:", OUTPUT_DIR)
print("=" * 80)

# Start Streamlit server in the background
port = 8503
app_url = f"http://localhost:{port}"

print(f"\n[1/3] Starting Streamlit server on port {port}...")
server_process = subprocess.Popen(
    [
        PYTHON_EXE,
        "-m",
        "streamlit",
        "run",
        "app.py",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.serverAddress=localhost",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ],
    cwd=str(WORKSPACE_DIR),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(12)  # Wait for Streamlit and model bundle to initialize

try:
    print("\n[2/3] Launching headless browser to capture screenshots...")
    with sync_playwright() as p:
        # Launch using system Microsoft Edge
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--window-size=1920,1080"]
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        page = context.new_page()

        print(f"  * Navigating to {app_url}...")
        page.goto(app_url, wait_until="networkidle", timeout=60000)
        time.sleep(6)  # Wait for Folium map tiles to render

        # ----------------------------------------------------------------------
        # Figure 13: Main dashboard and study area map
        # ----------------------------------------------------------------------
        print("  * Capturing Figure 13 – Main dashboard and study area map...")
        fig13_path = OUTPUT_DIR / "Figure_13_Main_Dashboard_and_Study_Area_Map.png"
        page.screenshot(path=str(fig13_path), full_page=False)
        print(f"    -> Saved {fig13_path.name}")

        # ----------------------------------------------------------------------
        # Select a representative central grid (Grid 1 or click select button)
        # ----------------------------------------------------------------------
        print("  * Selecting representative grid for Figures 14, 15, 16, 18...")
        try:
            # Click "Select this grid" button in sidebar
            select_btn = page.locator("button:has-text('Select this grid')")
            if select_btn.count() > 0:
                select_btn.first.click()
                time.sleep(5)  # wait for grid panel to load
        except Exception as e:
            print("    Notice clicking select:", e)

        # ----------------------------------------------------------------------
        # Figure 14: Interactive grid selection and information panel
        # ----------------------------------------------------------------------
        print("  * Capturing Figure 14 – Interactive grid selection and info panel...")
        fig14_path = OUTPUT_DIR / "Figure_14_Interactive_Grid_Selection_and_Info_Panel.png"
        page.screenshot(path=str(fig14_path), full_page=False)
        print(f"    -> Saved {fig14_path.name}")

        # ----------------------------------------------------------------------
        # Figure 15: Urban activity prediction and 24-hour pulse
        # ----------------------------------------------------------------------
        print("  * Capturing Figure 15 – Urban activity prediction and 24-hour pulse...")
        # Scroll slightly to bring 24-hour pulse chart into focus
        page.evaluate("window.scrollBy(0, 350)")
        time.sleep(2)
        fig15_path = OUTPUT_DIR / "Figure_15_Urban_Activity_Prediction_and_24Hour_Pulse.png"
        page.screenshot(path=str(fig15_path), full_page=False)
        print(f"    -> Saved {fig15_path.name}")

        # ----------------------------------------------------------------------
        # Figure 16: SHAP explanation and AI-generated response
        # ----------------------------------------------------------------------
        print("  * Capturing Figure 16 – SHAP explanation & AI response...")
        # Expand SHAP accordion if collapsed
        try:
            shap_exp = page.locator("text=SHAP Explanation")
            if shap_exp.count() > 0:
                shap_exp.first.click()
                time.sleep(2)
        except Exception:
            pass

        # Click a suggestion button to generate an AI response
        try:
            sugg_btn = page.locator("button:has-text('Why did the model make this prediction?')")
            if sugg_btn.count() > 0:
                sugg_btn.first.click()
                time.sleep(8)  # wait for LLM/fallback agent response
        except Exception as e:
            print("    Notice on AI question trigger:", e)

        page.evaluate("window.scrollBy(0, 500)")
        time.sleep(2)
        fig16_path = OUTPUT_DIR / "Figure_16_SHAP_Explanation_and_AI_Response.png"
        page.screenshot(path=str(fig16_path), full_page=False)
        print(f"    -> Saved {fig16_path.name}")

        # ----------------------------------------------------------------------
        # Figure 18: What-if scenario analysis
        # ----------------------------------------------------------------------
        print("  * Capturing Figure 18 – What-if scenario analysis...")
        try:
            wi_exp = page.locator("text=What-If Scenario")
            if wi_exp.count() > 0:
                wi_exp.first.click()
                time.sleep(2)
            # Click Run What-If Scenario button
            wi_btn = page.locator("button:has-text('Run What-If Scenario')")
            if wi_btn.count() > 0:
                wi_btn.first.click()
                time.sleep(4)
        except Exception as e:
            print("    Notice on What-If trigger:", e)

        fig18_path = OUTPUT_DIR / "Figure_18_WhatIf_Scenario_Analysis.png"
        page.screenshot(path=str(fig18_path), full_page=False)
        print(f"    -> Saved {fig18_path.name}")

        # ----------------------------------------------------------------------
        # Figure 19: Multi-agent workflow or execution log
        # ----------------------------------------------------------------------
        print("  * Capturing Figure 19 – Multi-agent workflow & execution log...")
        try:
            # Enable agent debug log toggle in sidebar
            toggle = page.locator("text=Show agent debug log")
            if toggle.count() > 0:
                toggle.first.click()
                time.sleep(2)
            debug_exp = page.locator("text=Agent Debug Log")
            if debug_exp.count() > 0:
                debug_exp.first.click()
                time.sleep(2)
        except Exception as e:
            print("    Notice on Debug Log toggle:", e)

        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        fig19_path = OUTPUT_DIR / "Figure_19_MultiAgent_Workflow_Execution_Log.png"
        page.screenshot(path=str(fig19_path), full_page=False)
        print(f"    -> Saved {fig19_path.name}")

        # ----------------------------------------------------------------------
        # Figure 17: Theory-guided interpretation tab (Mode 4 TSE-UI)
        # ----------------------------------------------------------------------
        print("  * Capturing Figure 17 – Theory-guided interpretation tab (Mode 4)...")
        # Select Mode 4 radio in sidebar
        try:
            tse_radio = page.locator("text=Scientific Discovery & Evolution")
            if tse_radio.count() > 0:
                tse_radio.first.click()
                time.sleep(6)  # wait for TSE-UI tab view to render
        except Exception as e:
            print("    Notice on TSE-UI switch:", e)

        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)
        fig17_path = OUTPUT_DIR / "Figure_17_Theory_Guided_Interpretation_TSE_UI.png"
        page.screenshot(path=str(fig17_path), full_page=False)
        print(f"    -> Saved {fig17_path.name}")

        browser.close()
        print("\n[3/3] All screenshots captured successfully!")

finally:
    print("  * Shutting down background Streamlit server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except Exception:
        server_process.kill()

print("=" * 80)
print("SUCCESS: Figures 13 - 19 captured and saved to:", OUTPUT_DIR)
print("=" * 80)

