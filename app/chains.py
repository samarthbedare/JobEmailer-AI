import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Samarth Narendra Bedare, a job applicant interested in the role mentioned above. You are applying to a company for a position 
        that aligns with your skills and experiences. Write an email expressing your interest in the job, showcasing your background 
        and how your skills can contribute to the company's goals. 
        Mention relevant experiences and achievements that make you a strong candidate for this position. 
        Additionally, include the following links to certifications that demonstrate your proficiency in the skills required for the role:
        {link_list}

        ### CANDIDATE DETAILS:
        - **Name**: Samarth Narendra Bedare
        - **Email**: bedaresamarth@gmail.com
        - **Phone**: (+91) 8793634732
        - **Skills**: Java, Python, C#, C++, MySQL, MS SQL Server, MS Visual Studio, Next.js, HTML, CSS, Javascript, MERN, Flask, Django, Web Development, Android, Desktop App Development, AI, ML, Deep Learning, Teamwork and Collaboration, Leadership and Project Management, Time Management, Good Communication.
        - **Education**:
            - **B.E in Computer Engineering** | Dr. D. Y. Patil Institute of Technology, Pimpri, Pune (Pursuing, Third Year, CGPA: 9.52, 2023-2026).
            - **Diploma in Computer Engineering** | Marathwada Mitra Mandal’s Polytechnic, Pune (Final Year Percentage: 91.31%, 2020-2023).
        - **Academic Projects**:
            - **Automated Fracture Detection System**: Developed a machine learning model with 90%+ accuracy for automated fracture detection in X-rays, reducing manual work by 80% and improving diagnostic efficiency by 40%. Role: Developer.
            - **Blog Full Stack Website**: Developed a blog website using the Next.js-enhanced MERN stack, enabling 100+ users for seamless content creation. Role: Developer. Link: [newspiral.vercel.app](http://newspiral.vercel.app).
            - **College Voting Portal**: Developed a secure desktop app for online college voting. Role: Team Lead & Developer. Technologies: MS Visual Studio, MS SQL Server.
        - **Position of Responsibility**:
            - **Technical Head** | Dr. D. Y. Patil Institute of Technology (Aug 2024).
            - **R&D Club Member** | Dr. D. Y. Patil Institute of Technology (Aug 2024 - Present).
        - **Extracurricular / Achievements**:
            - 1st rank in Diploma with 91.31%.
            - IBM Machine Learning Professional Certificate on Coursera (In Progress).
            - Java Data Structures & Algorithms Certificate on Udemy (Completion Date: June 2024).
            - Accenture NA Coding: Development & Advanced Engineering Job Simulation – (Nov 2024).
            - Published Research Paper on “College Voting Portal” (IRJMETS).
            - 3 ongoing Copyrights and Patents.
        - **Social Links**:
            - **GitHub**: [github.com/samarthbedare](https://github.com/samarthbedare)
            - **LinkedIn**: [linkedin.com/in/samarthbedare](https://linkedin.com/in/samarthbedare)

        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))