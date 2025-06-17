#!/usr/bin/env python3
"""
Test Runner Script for Bank Churn Analysis Project

This script provides a unified interface for running different types of tests
with various configurations and reporting options.

Usage:
    python run_tests.py --unit                    # Run unit tests only
    python run_tests.py --integration             # Run integration tests only
    python run_tests.py --api                     # Run API tests only
    python run_tests.py --all                     # Run all tests
    python run_tests.py --coverage                # Run with coverage report
    python run_tests.py --performance             # Run performance tests
    python run_tests.py --smoke                   # Run smoke tests
    python run_tests.py --parallel                # Run tests in parallel
    python run_tests.py --verbose                 # Verbose output
    python run_tests.py --quick                   # Quick test run (skip slow tests)

Author: Bank Churn Analysis Team
Date: 2024
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


class TestRunner:
    """Test runner for the bank churn analysis project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / 'src' / 'tests'
        self.results = {}
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        required_packages = [
            'pytest',
            'pytest-cov',
            'pytest-html',
            'pytest-xdist',
            'pandas',
            'numpy',
            'scikit-learn',
            'xgboost',
            'shap',
            'fastapi',
            'uvicorn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Please install them using: pip install -r requirements.txt")
            return False
        
        print("✅ All required dependencies are installed")
        return True
    
    def setup_test_environment(self) -> bool:
        """Setup the test environment."""
        try:
            # Create necessary directories
            test_dirs = [
                self.project_root / 'data' / 'raw',
                self.project_root / 'data' / 'interim',
                self.project_root / 'data' / 'processed',
                self.project_root / 'models',
                self.project_root / 'reports' / 'figures',
                self.project_root / 'reports' / 'test_results'
            ]
            
            for test_dir in test_dirs:
                test_dir.mkdir(parents=True, exist_ok=True)
            
            # Set environment variables for testing
            os.environ['TESTING'] = '1'
            os.environ['PYTHONPATH'] = str(self.project_root / 'src')
            
            print("✅ Test environment setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup test environment: {e}")
            return False
    
    def run_pytest(self, test_args: List[str], test_name: str) -> Dict:
        """Run pytest with given arguments and return results."""
        print(f"\n🧪 Running {test_name}...")
        print(f"Command: pytest {' '.join(test_args)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest'] + test_args,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse pytest output for test counts
            output_lines = result.stdout.split('\n')
            summary_line = None
            for line in reversed(output_lines):
                if 'passed' in line or 'failed' in line or 'error' in line:
                    summary_line = line
                    break
            
            test_result = {
                'name': test_name,
                'returncode': result.returncode,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'summary': summary_line or 'No summary available',
                'success': result.returncode == 0
            }
            
            if test_result['success']:
                print(f"✅ {test_name} completed successfully in {duration:.2f}s")
            else:
                print(f"❌ {test_name} failed in {duration:.2f}s")
                if result.stderr:
                    print(f"Error output: {result.stderr[:500]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} timed out after 30 minutes")
            return {
                'name': test_name,
                'returncode': -1,
                'duration': 1800,
                'stdout': '',
                'stderr': 'Test timed out',
                'summary': 'Timed out',
                'success': False
            }
        except Exception as e:
            print(f"❌ Error running {test_name}: {e}")
            return {
                'name': test_name,
                'returncode': -1,
                'duration': 0,
                'stdout': '',
                'stderr': str(e),
                'summary': f'Error: {e}',
                'success': False
            }
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = False) -> Dict:
        """Run unit tests."""
        args = [
            str(self.test_dir / 'test_data_pipeline.py'),
            str(self.test_dir / 'test_models.py'),
            '-v' if verbose else '-q',
            '--tb=short',
            '--disable-warnings',
            '--maxfail=1'
        ]
        
        if coverage:
            args.extend(['--cov'])
        
        return self.run_pytest(args, "Unit Tests")
    
    def run_api_tests(self, verbose: bool = False, coverage: bool = False) -> Dict:
        """Run API tests."""
        args = [
            str(self.test_dir / 'test_api.py'),
            '-v' if verbose else '-q',
            '--tb=short',
            '--disable-warnings',
            '--maxfail=1'
        ]
        
        if coverage:
            args.extend(['--cov'])
        
        return self.run_pytest(args, "API Tests")
    
    def run_integration_tests(self, verbose: bool = False, coverage: bool = False) -> Dict:
        """Run integration tests."""
        args = [
            str(self.test_dir / 'test_integration.py'),
            '-v' if verbose else '-q',
            '--tb=short',
            '--runintegration'
        ]
        
        if coverage:
            args.extend([
                '--cov=src',
                '--cov-report=html:reports/test_results/coverage_integration',
                '--cov-report=term-missing'
            ])
        
        return self.run_pytest(args, "Integration Tests")
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = False, 
                     fast: bool = False) -> Dict[str, Dict]:
        """Run all test suites."""
        args = [
            str(self.test_dir),
            '-v' if verbose else '-q',
            '--tb=short',
            '--disable-warnings'
        ]
        
        if fast:
            args.extend(['-m', 'not slow', '--maxfail=1'])
        else:
            args.extend(['--maxfail=5'])
        
        if coverage:
            # Coverage settings are now in pytest.ini
            args.extend(['--cov'])
        
        return self.run_pytest(args, "All Tests")
    
    def run_performance_tests(self, verbose: bool = False) -> Dict:
        """Run performance tests."""
        args = [
            str(self.test_dir),
            '-v' if verbose else '-q',
            '--tb=short',
            '-m', 'slow',
            '--runslow'
        ]
        
        return self.run_pytest(args, "Performance Tests")
    
    def generate_test_report(self, results: Dict[str, Dict]) -> None:
        """Generate a comprehensive test report."""
        report_dir = self.project_root / 'reports' / 'test_results'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / 'test_summary.txt'
        
        with open(report_file, 'w') as f:
            f.write("Bank Churn Analysis - Test Execution Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test execution completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_duration = 0
            total_success = 0
            total_tests = len(results)
            
            for test_name, result in results.items():
                f.write(f"Test Suite: {test_name}\n")
                f.write(f"Status: {'✅ PASSED' if result['success'] else '❌ FAILED'}\n")
                f.write(f"Duration: {result['duration']:.2f} seconds\n")
                f.write(f"Summary: {result['summary']}\n")
                
                if not result['success'] and result['stderr']:
                    f.write(f"Error Details: {result['stderr'][:1000]}\n")
                
                f.write("-" * 30 + "\n\n")
                
                total_duration += result['duration']
                if result['success']:
                    total_success += 1
            
            f.write(f"Overall Summary:\n")
            f.write(f"Total test suites: {total_tests}\n")
            f.write(f"Successful: {total_success}\n")
            f.write(f"Failed: {total_tests - total_success}\n")
            f.write(f"Total duration: {total_duration:.2f} seconds\n")
            f.write(f"Success rate: {(total_success/total_tests)*100:.1f}%\n")
        
        print(f"\n📊 Test report generated: {report_file}")
    
    def print_summary(self, results: Dict[str, Dict]) -> None:
        """Print test execution summary."""
        print("\n" + "=" * 60)
        print("🧪 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        total_duration = 0
        successful_tests = 0
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            duration = result['duration']
            print(f"{test_name:20} | {status:10} | {duration:6.2f}s | {result['summary']}")
            
            total_duration += duration
            if result['success']:
                successful_tests += 1
        
        print("-" * 60)
        print(f"Total Tests: {len(results)} | Passed: {successful_tests} | Failed: {len(results) - successful_tests}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Success Rate: {(successful_tests/len(results))*100:.1f}%")
        
        if successful_tests == len(results):
            print("\n🎉 All tests passed! The churn analysis system is ready for deployment.")
        else:
            print(f"\n⚠️  {len(results) - successful_tests} test suite(s) failed. Please review and fix issues.")
        
        print("=" * 60)


def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test runner for Bank Churn Analysis project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit --coverage  # Run unit tests with coverage
  python run_tests.py --integration -v   # Run integration tests verbosely
  python run_tests.py --api --fast       # Run API tests, skip slow ones
  python run_tests.py --performance      # Run only performance tests
        """
    )
    
    # Test selection arguments
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--unit', action='store_true', help='Run only unit tests')
    test_group.add_argument('--api', action='store_true', help='Run only API tests')
    test_group.add_argument('--integration', action='store_true', help='Run only integration tests')
    test_group.add_argument('--performance', action='store_true', help='Run only performance tests')
    test_group.add_argument('--all', action='store_true', default=True, help='Run all tests (default)')
    
    # Test execution options
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-c', '--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('-f', '--fast', action='store_true', help='Skip slow tests')
    parser.add_argument('--no-report', action='store_true', help='Skip generating test report')
    parser.add_argument('--check-deps', action='store_true', help='Only check dependencies')
    
    args = parser.parse_args()
    
    # Initialize test runner
    project_root = Path(__file__).parent
    runner = TestRunner(project_root)
    
    print("🚀 Bank Churn Analysis - Test Runner")
    print("=" * 40)
    
    # Check dependencies
    if not runner.check_dependencies():
        sys.exit(1)
    
    if args.check_deps:
        print("✅ Dependency check completed successfully")
        sys.exit(0)
    
    # Setup test environment
    if not runner.setup_test_environment():
        sys.exit(1)
    
    # Run tests based on arguments
    results = {}
    
    try:
        if args.unit:
            results['Unit Tests'] = runner.run_unit_tests(args.verbose, args.coverage)
        elif args.api:
            results['API Tests'] = runner.run_api_tests(args.verbose, args.coverage)
        elif args.integration:
            results['Integration Tests'] = runner.run_integration_tests(args.verbose, args.coverage)
        elif args.performance:
            results['Performance Tests'] = runner.run_performance_tests(args.verbose)
        else:
            # Run all tests
            result = runner.run_all_tests(args.verbose, args.coverage, args.fast)
            results['All Tests'] = result
        
        # Print summary
        runner.print_summary(results)
        
        # Generate detailed report
        if not args.no_report:
            runner.generate_test_report(results)
        
        # Exit with appropriate code
        failed_tests = sum(1 for result in results.values() if not result['success'])
        sys.exit(failed_tests)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error during test execution: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()