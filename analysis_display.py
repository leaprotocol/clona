class AnalysisDisplay:
    def __init__(self):
        self.scenario_details = None

    def show_analysis_results(self, analysis, config):
        """Display analysis results based on configuration"""
        with ui.card().classes('w-full p-4 mb-4'):
            ui.label(config['title']).classes('text-xl font-bold mb-2')
            self.display_metrics(analysis, config['metrics'])
            if config.get('show_visualization'):
                self.show_visualization(analysis) 