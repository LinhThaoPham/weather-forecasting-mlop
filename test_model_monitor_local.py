"""
Local debugging script to test Model Monitor functionality.
Captures screenshots and console logs to identify the bug.
"""
import sys
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
import time

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_model_monitor():
    """Test Model Monitor button and endpoint responses."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # visible for debugging
        page = browser.new_page()
        
        # Setup console log capture
        console_logs = []
        page.on("console", lambda msg: console_logs.append({
            "type": msg.type,
            "text": msg.text,
            "location": msg.location
        }))
        
        # Navigate to dashboard
        print("🌐 Navigating to http://localhost:5173...")
        page.goto("http://localhost:5173", wait_until="networkidle", timeout=30000)
        print("✓ Dashboard loaded")
        
        # Take initial screenshot
        page.screenshot(path="/tmp/dashboard_initial.png", full_page=True)
        print("📸 Initial screenshot saved: /tmp/dashboard_initial.png")
        
        # Check if Model Monitor button exists
        try:
            monitor_button = page.locator("#nav-monitor, button:has-text('Model Monitor'), [data-testid='model-monitor']")
            if monitor_button.count() > 0:
                print("✓ Found Model Monitor button")
                page.screenshot(path="/tmp/before_click.png", full_page=True)
                
                # Try to click the button
                print("🖱️  Clicking Model Monitor button...")
                monitor_button.first.click()
                time.sleep(2)
                
                # Take screenshot after click
                page.screenshot(path="/tmp/after_click.png", full_page=True)
                print("📸 After-click screenshot saved: /tmp/after_click.png")
                
                # Check for Model Monitor section visibility
                monitor_section = page.locator("#modelMonitorSection")
                if monitor_section.count() > 0:
                    is_visible = monitor_section.is_visible()
                    print(f"📊 Model Monitor section visible: {is_visible}")
                else:
                    print("⚠️  Model Monitor section not found in DOM")
                
            else:
                print("❌ Model Monitor button not found")
        except Exception as e:
            print(f"❌ Error finding/clicking button: {e}")
        
        # Test data-api endpoints directly
        print("\n📡 Testing data-api endpoints...")
        
        # Test /model/registry
        try:
            registry_response = page.evaluate("""
                async () => {
                    try {
                        const resp = await fetch('http://127.0.0.1:8001/model/registry');
                        const data = await resp.json();
                        return { status: resp.status, data: data };
                    } catch (e) {
                        return { error: e.message };
                    }
                }
            """)
            print(f"  /model/registry: {registry_response}")
        except Exception as e:
            print(f"  ❌ /model/registry error: {e}")
        
        # Test /model/history
        try:
            history_response = page.evaluate("""
                async () => {
                    try {
                        const resp = await fetch('http://127.0.0.1:8001/model/history');
                        const data = await resp.json();
                        return { status: resp.status, data: data };
                    } catch (e) {
                        return { error: e.message };
                    }
                }
            """)
            print(f"  /model/history: {history_response}")
        except Exception as e:
            print(f"  ❌ /model/history error: {e}")
        
        # Print captured console logs
        print("\n📋 Console logs captured:")
        for log in console_logs:
            print(f"  [{log['type'].upper()}] {log['text']}")
        
        # Dump page content for inspection
        content = page.content()
        with open("/tmp/page_content.html", "w", encoding="utf-8") as f:
            f.write(content)
        print("\n💾 Full page content saved: /tmp/page_content.html")
        
        browser.close()
        print("\n✓ Test complete")

if __name__ == "__main__":
    test_model_monitor()
