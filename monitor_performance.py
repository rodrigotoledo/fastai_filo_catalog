#!/usr/bin/env python3
"""
Performance Monitor for Photo Search Service
Phase 4: Monitoring - Automated health checks and alerting
"""

import requests
import json
import time
import logging
from datetime import datetime
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor photo search service performance and health"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.metrics_endpoint = f"{self.base_url}/api/v1/photos/metrics"
        self.health_endpoint = f"{self.base_url}/api/v1/photos/health"

    def check_health(self) -> Dict[str, Any]:
        """Check service health and return status"""
        try:
            response = requests.get(self.health_endpoint, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unreachable",
                "error": str(e),
                "timestamp": time.time()
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        try:
            response = requests.get(self.metrics_endpoint, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Metrics collection failed: {e}")
            return {
                "status": "unreachable",
                "error": str(e),
                "timestamp": time.time()
            }

    def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics and provide insights"""
        analysis = {
            "performance_score": 100,  # Start with perfect score
            "recommendations": [],
            "insights": []
        }

        if "metrics" in metrics:
            m = metrics["metrics"]

            # Analyze search performance
            avg_time = m.get("avg_search_time_ms", 0)
            if avg_time > 1000:
                analysis["performance_score"] -= 30
                analysis["recommendations"].append("Average search time > 1s - consider cache optimization")
            elif avg_time > 500:
                analysis["performance_score"] -= 15
                analysis["recommendations"].append("Average search time > 500ms - monitor closely")

            # Analyze cache performance
            cache_hit_rate = m.get("cache_hit_rate_percent", 0)
            if cache_hit_rate < 70:
                analysis["performance_score"] -= 20
                analysis["recommendations"].append(f"Cache hit rate {cache_hit_rate}% < 70% - review cache strategy")
            elif cache_hit_rate < 80:
                analysis["performance_score"] -= 10
                analysis["recommendations"].append(f"Cache hit rate {cache_hit_rate}% could be higher")

            # Analyze re-ranking usage
            reranking_rate = m.get("reranking_rate_percent", 0)
            if reranking_rate > 50:
                analysis["insights"].append(f"High re-ranking usage ({reranking_rate}%) - complex queries detected")
            elif reranking_rate < 10:
                analysis["insights"].append(f"Low re-ranking usage ({reranking_rate}%) - mostly simple queries")

            # Analyze query distribution
            query_types = m.get("query_types", {})
            total_queries = sum(query_types.values())
            if total_queries > 0:
                complex_pct = (query_types.get("complex", 0) + query_types.get("question", 0)) / total_queries * 100
                if complex_pct > 30:
                    analysis["insights"].append(f"High complex query rate ({complex_pct:.1f}%) - LLM working hard")

        analysis["performance_score"] = max(0, analysis["performance_score"])
        return analysis

    def generate_report(self) -> str:
        """Generate a comprehensive performance report"""
        logger.info("🔍 Generating performance report...")

        health = self.check_health()
        metrics = self.get_metrics()
        analysis = self.analyze_performance(metrics)

        report = f"""
📊 PHOTO SEARCH PERFORMANCE REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔴 SERVICE HEALTH
Status: {health.get('status', 'unknown').upper()}
Uptime: {health.get('uptime_seconds', 0):.0f} seconds
Alerts: {health.get('alerts', {}).get('alerts_count', 0)}

⚡ PERFORMANCE METRICS
"""

        if "metrics" in metrics:
            m = metrics["metrics"]
            report += f"""Total Searches: {m.get('total_searches', 0)}
Searches/sec: {m.get('searches_per_second', 0):.2f}
Avg Response Time: {m.get('avg_search_time_ms', 0):.1f}ms
Cache Hit Rate: {m.get('cache_hit_rate_percent', 0):.1f}%
Re-ranking Rate: {m.get('reranking_rate_percent', 0):.1f}%
Slow Queries: {m.get('slow_queries_count', 0)}

📈 QUERY DISTRIBUTION
"""
            query_types = m.get("query_types", {})
            for qtype, count in query_types.items():
                report += f"{qtype.capitalize()}: {count}\n"

        report += f"""
🎯 PERFORMANCE SCORE: {analysis['performance_score']}/100

💡 INSIGHTS
"""
        for insight in analysis["insights"]:
            report += f"• {insight}\n"

        if analysis["recommendations"]:
            report += f"""
⚠️  RECOMMENDATIONS
"""
            for rec in analysis["recommendations"]:
                report += f"• {rec}\n"

        # Add alerts if any
        if health.get("alerts", {}).get("alerts"):
            report += f"""
🚨 ACTIVE ALERTS
"""
            for alert in health["alerts"]["alerts"]:
                report += f"[{alert['level'].upper()}] {alert['message']}\n"

        return report

def main():
    """Main monitoring function"""
    monitor = PerformanceMonitor()

    print("🚀 Photo Search Performance Monitor")
    print("=" * 50)

    # Generate and display report
    report = monitor.generate_report()
    print(report)

    # Save report to file
    report_file = "performance_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved to {report_file}")

if __name__ == "__main__":
    main()
