"""
Recommendation Engine for HC-SmartPulse
Provides actionable HR recommendations based on prediction results and employee features
"""

import pandas as pd
from typing import List, Dict, Tuple


class RecommendationEngine:
    """Generate personalized HR recommendations based on attrition risk and employee profile"""
    
    def __init__(self, risk_threshold=0.7):
        """
        Initialize recommendation engine
        
        Args:
            risk_threshold: Probability threshold to consider an employee as high risk
        """
        self.risk_threshold = risk_threshold
    
    def get_recommendations(self, 
                           risk_probability: float,
                           employee_data: Dict) -> List[Dict[str, str]]:
        """
        Generate recommendations based on risk probability and employee features
        
        Args:
            risk_probability: Predicted attrition probability (0-1)
            employee_data: Dictionary containing employee features
            
        Returns:
            List of recommendation dictionaries with 'category', 'priority', and 'action' keys
        """
        recommendations = []
        
        # Only provide recommendations for high-risk employees
        if risk_probability < self.risk_threshold:
            return [{
                'category': 'Low Risk',
                'priority': 'Monitor',
                'action': 'Employee shows low flight risk. Continue standard engagement practices.'
            }]
        
        # Priority counter for sorting
        priority_map = {'Critical': 1, 'High': 2, 'Medium': 3}
        
        # Rule 1: Overtime workload management
        if employee_data.get('OverTime', 0) == 1 or employee_data.get('OverTime', 'No') == 'Yes':
            recommendations.append({
                'category': 'Workload Management',
                'priority': 'Critical',
                'action': 'Rekomendasi: Peninjauan ulang beban kerja (Workload Re-balancing). '
                         'Pertimbangkan untuk mendistribusikan tugas secara lebih merata atau '
                         'memberikan kompensasi lembur yang lebih kompetitif untuk mencegah burnout.'
            })
        
        # Rule 2: Career development and promotion
        years_since_promotion = employee_data.get('YearsSinceLastPromotion', 0)
        if years_since_promotion >= 3:
            recommendations.append({
                'category': 'Career Development',
                'priority': 'Critical',
                'action': f'Rekomendasi: Review jalur karier (Career Path Review). '
                         f'Karyawan tidak mendapat promosi selama {years_since_promotion} tahun. '
                         f'Tawarkan program upskilling, mentoring, atau jalur promosi yang jelas '
                         f'untuk meningkatkan engagement dan retensi.'
            })
        
        # Rule 3: Environment satisfaction
        env_satisfaction = employee_data.get('EnvironmentSatisfaction', 3)
        if env_satisfaction <= 2:  # Low satisfaction (1-2 on scale)
            recommendations.append({
                'category': 'Workplace Environment',
                'priority': 'High',
                'action': 'Rekomendasi: Intervensi manajerial atau sesi feedback 1-on-1. '
                         'Tingkat kepuasan lingkungan kerja rendah. Lakukan investigasi akar masalah '
                         'melalui sesi feedback personal dan perbaiki iklim kerja tim.'
            })
        
        # Rule 4: Job satisfaction
        job_satisfaction = employee_data.get('JobSatisfaction', 3)
        if job_satisfaction <= 2:
            recommendations.append({
                'category': 'Job Satisfaction',
                'priority': 'High',
                'action': 'Rekomendasi: Job Redesign atau Role Enrichment. '
                         'Kepuasan kerja rendah mengindikasikan potensi job misfit. '
                         'Pertimbangkan untuk merestrukturisasi tanggung jawab atau rotasi peran.'
            })
        
        # Rule 5: Work-life balance
        work_life_balance = employee_data.get('WorkLifeBalance', 3)
        if work_life_balance <= 2:
            recommendations.append({
                'category': 'Work-Life Balance',
                'priority': 'High',
                'action': 'Rekomendasi: Flexible Work Arrangements. '
                         'Work-life balance rendah. Tawarkan opsi remote work, flexible hours, '
                         'atau program wellness untuk meningkatkan keseimbangan hidup-kerja.'
            })
        
        # Rule 6: Distance from home
        distance_from_home = employee_data.get('DistanceFromHome', 0)
        if distance_from_home > 20:  # Long commute
            recommendations.append({
                'category': 'Commute Support',
                'priority': 'Medium',
                'action': f'Rekomendasi: Relokasi atau Remote Work Support. '
                         f'Jarak rumah-kantor ({distance_from_home} km) cukup jauh. '
                         f'Pertimbangkan opsi remote/hybrid work atau bantuan relokasi.'
            })
        
        # Rule 7: Salary competitiveness
        monthly_income = employee_data.get('MonthlyIncome', 0)
        job_level = employee_data.get('JobLevel', 1)
        
        # Rough benchmark for low salary (this can be adjusted)
        if job_level >= 2 and monthly_income < 5000:
            recommendations.append({
                'category': 'Compensation Review',
                'priority': 'Critical',
                'action': 'Rekomendasi: Salary Benchmarking dan Adjustment. '
                         'Kompensasi mungkin tidak kompetitif untuk level jabatan. '
                         'Lakukan benchmarking eksternal dan pertimbangkan penyesuaian gaji '
                         'atau bonus retensi.'
            })
        
        # Rule 8: Training opportunities
        training_times_last_year = employee_data.get('TrainingTimesLastYear', 0)
        if training_times_last_year == 0:
            recommendations.append({
                'category': 'Professional Development',
                'priority': 'Medium',
                'action': 'Rekomendasi: Increase Training Investment. '
                         'Tidak ada pelatihan dalam setahun terakhir. '
                         'Sediakan program development dan sertifikasi profesional '
                         'untuk meningkatkan skills dan motivasi.'
            })
        
        # Rule 9: Relationship with manager
        relationship_satisfaction = employee_data.get('RelationshipSatisfaction', 3)
        if relationship_satisfaction <= 2:
            recommendations.append({
                'category': 'Manager Relationship',
                'priority': 'High',
                'action': 'Rekomendasi: Leadership Coaching & Mentoring. '
                         'Hubungan dengan atasan kurang baik. '
                         'Fasilitasi sesi coaching untuk manager dan employee untuk '
                         'memperbaiki komunikasi dan relationship.'
            })
        
        # Sort recommendations by priority
        recommendations.sort(key=lambda x: priority_map.get(x['priority'], 4))
        
        # If no specific recommendations, add general one
        if not recommendations:
            recommendations.append({
                'category': 'General Retention',
                'priority': 'High',
                'action': 'Rekomendasi: Comprehensive Retention Interview. '
                         'Karyawan berisiko tinggi namun tidak ada flag spesifik terdeteksi. '
                         'Lakukan retention interview mendalam untuk memahami concern utama.'
            })
        
        return recommendations
    
    def get_risk_level(self, risk_probability: float) -> Tuple[str, str]:
        """
        Categorize risk level based on probability
        
        Args:
            risk_probability: Predicted attrition probability (0-1)
            
        Returns:
            Tuple of (risk_level, color_code)
        """
        if risk_probability >= 0.7:
            return ('High Risk', 'red')
        elif risk_probability >= 0.4:
            return ('Medium Risk', 'orange')
        else:
            return ('Low Risk', 'green')
    
    def format_recommendations_text(self, recommendations: List[Dict[str, str]]) -> str:
        """
        Format recommendations as readable text
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Formatted string with all recommendations
        """
        if not recommendations:
            return "No specific recommendations available."
        
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(
                f"**{i}. [{rec['priority']}] {rec['category']}**\n"
                f"{rec['action']}\n"
            )
        
        return "\n".join(formatted)


def test_recommendation_engine():
    """Test the recommendation engine with sample data"""
    engine = RecommendationEngine(risk_threshold=0.7)
    
    # Test case 1: High risk with overtime
    print("Test Case 1: High Risk + Overtime")
    print("=" * 60)
    employee1 = {
        'OverTime': 'Yes',
        'YearsSinceLastPromotion': 5,
        'EnvironmentSatisfaction': 1,
        'JobSatisfaction': 2,
        'MonthlyIncome': 3000,
        'JobLevel': 2
    }
    
    recommendations = engine.get_recommendations(0.85, employee1)
    print(engine.format_recommendations_text(recommendations))
    
    # Test case 2: Low risk
    print("\n\nTest Case 2: Low Risk")
    print("=" * 60)
    employee2 = {
        'OverTime': 'No',
        'YearsSinceLastPromotion': 1,
        'EnvironmentSatisfaction': 4,
        'JobSatisfaction': 4,
        'MonthlyIncome': 8000,
        'JobLevel': 3
    }
    
    recommendations = engine.get_recommendations(0.25, employee2)
    print(engine.format_recommendations_text(recommendations))


if __name__ == "__main__":
    test_recommendation_engine()
