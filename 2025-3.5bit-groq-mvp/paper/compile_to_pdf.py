#!/usr/bin/env python3
"""
Compile LaTeX to PDF using online LaTeX compilation service
"""

import requests
import sys
import os

def compile_latex_online(tex_file, output_pdf):
    """
    Compile LaTeX file to PDF using LaTeX.Online service
    """
    print(f"Reading {tex_file}...")

    if not os.path.exists(tex_file):
        print(f"Error: {tex_file} not found")
        return False

    with open(tex_file, 'r', encoding='utf-8') as f:
        latex_content = f.read()

    print(f"Compiling {tex_file} to PDF using online service...")
    print("This may take 30-60 seconds...")

    # Try LaTeX.Online service
    try:
        # Method 1: Direct POST
        url = "https://latexonline.cc/compile"

        files = {
            'file': (os.path.basename(tex_file), latex_content, 'text/plain')
        }

        print(f"Sending request to {url}...")
        response = requests.post(url, files=files, timeout=120)

        if response.status_code == 200:
            # Check if response is PDF
            if response.headers.get('content-type') == 'application/pdf':
                with open(output_pdf, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Success! PDF saved to: {output_pdf}")
                print(f"  File size: {len(response.content)} bytes")
                return True
            else:
                print(f"Warning: Response is not PDF")
                print(f"Content-Type: {response.headers.get('content-type')}")
                print(f"Response preview: {response.text[:500]}")
                return False
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except requests.exceptions.Timeout:
        print("Error: Request timed out (>120 seconds)")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def compile_with_texlive_net(tex_file, output_pdf):
    """
    Alternative: Try compiling with texlive.net
    """
    print(f"\nTrying alternative service: texlive.net...")

    with open(tex_file, 'r', encoding='utf-8') as f:
        latex_content = f.read()

    try:
        # texlive.net API endpoint
        url = "http://texlive.net/cgi-bin/latexcgi"

        files = {
            'filecontents': latex_content,
            'filename': os.path.basename(tex_file),
            'engine': 'pdflatex'
        }

        response = requests.post(url, data=files, timeout=120)

        if response.status_code == 200:
            with open(output_pdf, 'wb') as f:
                f.write(response.content)
            print(f"✓ Success! PDF saved to: {output_pdf}")
            return True
        else:
            print(f"Error: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """
    Main compilation function
    """
    print("="*60)
    print("LaTeX to PDF Online Compiler")
    print("="*60)

    # Compile main paper
    if compile_latex_online('paper.tex', 'paper.pdf'):
        print("\n✓ paper.pdf generated successfully!")
    else:
        print("\n✗ Failed to compile paper.tex")
        print("\nTrying alternative method...")
        if compile_with_texlive_net('paper.tex', 'paper.pdf'):
            print("\n✓ paper.pdf generated successfully (alternative method)!")
        else:
            print("\n✗ All compilation methods failed")
            print("\nPlease use Overleaf instead:")
            print("1. Go to https://www.overleaf.com")
            print("2. Upload paper_submission.zip")
            print("3. Download compiled PDF")
            return 1

    print("\n" + "="*60)

    # Compile supplementary (optional)
    if os.path.exists('supplementary.tex'):
        print("\nCompiling supplementary materials...")
        if compile_latex_online('supplementary.tex', 'supplementary.pdf'):
            print("✓ supplementary.pdf generated successfully!")
        else:
            print("✗ Failed to compile supplementary.tex (skipping)")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

    # List generated PDFs
    if os.path.exists('paper.pdf'):
        size = os.path.getsize('paper.pdf')
        print(f"\n✓ paper.pdf ({size:,} bytes)")
    if os.path.exists('supplementary.pdf'):
        size = os.path.getsize('supplementary.pdf')
        print(f"✓ supplementary.pdf ({size:,} bytes)")

    print("\nNext steps:")
    print("1. Open paper.pdf to review")
    print("2. Send to internal reviewers (Dec 19-23)")
    print("3. See INTERNAL_REVIEW_PACKAGE.md for email template")

    return 0

if __name__ == '__main__':
    sys.exit(main())
